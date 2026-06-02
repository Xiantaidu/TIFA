import json
import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup
from lib.config.io import load_raw_config
from lib.config.schema import ConfigurationScope


def _parse_int_list(ctx, param, value):
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]


def shared_options(func):
    options = [
        click.option(
            "--output-dir", "-o", required=True,
            type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
            help="Directory to save evaluation results.",
        ),
        click.option(
            "--ber-tolerance", type=float, default=50.0, show_default=True,
            help="Boundary error rate tolerance in milliseconds.",
        ),
        click.option(
            "--k-values", default="5,20", show_default=True,
            callback=_parse_int_list,
            help="Comma-separated worst-k values for per-token metrics.",
        ),
        click.option(
            "--conjunction-k", default="5,20", show_default=True,
            callback=_parse_int_list,
            help="Comma-separated worst-k values for pair conjunction metrics.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def online_options(func):
    options = [
        click.option(
            "--dataset", "-d", required=True,
            type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
            help="Path to the binarized dataset directory.",
        ),
        click.option(
            "--model", "-m", required=True,
            type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=pathlib.Path),
            help="Path to model checkpoint.",
        ),
        click.option(
            "--prefix", default="valid", show_default=True,
            help="Dataset prefix (e.g. valid, train).",
        ),
        click.option(
            "--batch-size", type=int, default=4, show_default=True,
            help="Batch size for evaluation.",
        ),
        click.option(
            "--num-workers", type=int, default=0, show_default=True,
            help="Number of dataloader worker processes.",
        ),
        click.option(
            "--precision", default="32-true", show_default=True,
            help="Precision for evaluation.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def offline_options(func):
    options = [
        click.option(
            "--pred-dir", "-p", required=True,
            type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
            help="Directory of predicted TextGrid files.",
        ),
        click.option(
            "--gt-dir", "-g", required=True,
            type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
            help="Directory of ground truth TextGrid files.",
        ),
        click.option(
            "--tier-name", default="phones", show_default=True,
            help="TextGrid tier name to extract.",
        ),
        click.option(
            "--stop-symbols", multiple=True, default=None,
            help="Stop symbols to filter from TextGrid intervals (repeatable).",
        ),
        click.option(
            "--mismatch-handling", default="raise",
            type=click.Choice(["raise", "skip"]), show_default=True,
            help="How to handle label mismatches: raise or skip.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def _check_file_and_config(file: pathlib.Path, expected):
    actual = load_raw_config(file, inherit=False, overrides=None)
    if actual != expected:
        raise RuntimeError(
            f"Contents of '{file}' do not match the model configuration. "
            f"The dataset was binarized with different parameters. "
            f"Please re-binarize."
        )


def _check_vocabulary(checkpoint_dir: pathlib.Path, dataset_dir: pathlib.Path):
    ckpt_vocab_path = checkpoint_dir / "vocabulary.json"
    dataset_vocab_path = dataset_dir / "vocabulary.json"
    if not ckpt_vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found at {ckpt_vocab_path}")
    if not dataset_vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found at {dataset_vocab_path}")
    with open(ckpt_vocab_path, "r", encoding="utf8") as f:
        ckpt_vocab = json.load(f)["symbols"]
    with open(dataset_vocab_path, "r", encoding="utf8") as f:
        dataset_vocab = json.load(f)["symbols"]
    if ckpt_vocab != dataset_vocab:
        raise RuntimeError(
            "Vocabulary mismatch between checkpoint and dataset. "
            "The dataset was binarized with a different vocabulary "
            "than the model was trained with."
        )


def _run_online_evaluation(
    scope: int,
    dataset: pathlib.Path,
    model: pathlib.Path,
    prefix: str,
    output_dir: pathlib.Path,
    batch_size: int,
    num_workers: int,
    precision: str,
    ber_tolerance: float,
    k_values: list[int],
    conjunction_k: list[int],
):
    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import (
        load_config_for_inference,
        load_inference_model,
        run_inference,
    )
    from inference.callbacks import EvaluationMetricsCallback
    from training.data import PhonemeTimingDataset

    backend, vocabulary, _ = load_inference_model(model, scope=scope)

    _, inference_config = load_config_for_inference(
        model.parent / "config.yaml", scope=scope,
    )
    _check_file_and_config(
        dataset / "feature.yaml", inference_config.features.model_dump()
    )
    _check_vocabulary(model.parent, dataset)

    feat = load_raw_config(dataset / "feature.yaml", inherit=False)
    unit_size_ms = feat["hop_size"] / feat["audio_sample_rate"] * 1000

    ds = PhonemeTimingDataset(
        data_dir=dataset,
        prefix=prefix,
        ensure_original_tokens=True,
        augmentation_config=None,
        return_waveform=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "summary.json"

    metric_callback = EvaluationMetricsCallback(
        unit_size_ms=unit_size_ms,
        vocab_size=vocabulary.vocab_size,
        ber_tolerance_ms=ber_tolerance,
        k_values=k_values,
        conjunction_k_values=conjunction_k,
        save_path=save_path,
    )

    run_inference(
        backend=backend,
        dataset=ds,
        callbacks=[metric_callback],
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        mode="evaluate",
    )
    logging.success("Online evaluation completed.", callback=rank_zero_info)


def _run_offline_evaluation(
    pred_dir: pathlib.Path,
    gt_dir: pathlib.Path,
    tier_name: str,
    stop_symbols: tuple[str, ...] | None,
    mismatch_handling: str,
    output_dir: pathlib.Path,
    ber_tolerance: float,
    k_values: list[int],
    conjunction_k: list[int],
):
    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import evaluate_offline
    from inference.callbacks import EvaluationMetricsCallback
    from inference.data import PairedDataset, TextGridDataset

    stop_set = set(stop_symbols) if stop_symbols else None

    pred_ds = TextGridDataset(pred_dir, tier_name=tier_name, stop_symbols=stop_set)
    gt_ds = TextGridDataset(gt_dir, tier_name=tier_name, stop_symbols=stop_set)

    paired = PairedDataset(
        pred_dataset=pred_ds,
        gt_dataset=gt_ds,
        mismatch_handling=mismatch_handling,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "summary.json"

    metric_callback = EvaluationMetricsCallback(
        unit_size_ms=1000,  # TextGrid times are in seconds -> ms
        vocab_size=paired.vocab_size,
        ber_tolerance_ms=ber_tolerance,
        k_values=k_values,
        conjunction_k_values=conjunction_k,
        save_path=save_path,
    )

    evaluate_offline(dataset=paired, callbacks=[metric_callback])
    logging.success("Offline evaluation completed.", callback=rank_zero_info)


@click.group(cls=DefaultGroup, help="Evaluate a model or compare TextGrids.")
def main():
    pass


@main.default_command()
@shared_options
@online_options
def supervised(**kwargs):
    """Supervised online evaluation on a binarized dataset."""
    _run_online_evaluation(scope=ConfigurationScope.FA, **kwargs)


@main.command(name="ssl")
@shared_options
@online_options
def ssl(**kwargs):
    """Self-supervised online evaluation."""
    _run_online_evaluation(scope=ConfigurationScope.FA_SSL, **kwargs)


@main.command(name="offline")
@shared_options
@offline_options
def offline(**kwargs):
    """Offline evaluation comparing two sets of TextGrid files."""
    _run_offline_evaluation(**kwargs)


if __name__ == "__main__":
    main()
