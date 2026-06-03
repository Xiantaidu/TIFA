import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup, csv_set
from lib.config.schema import ConfigurationScope


def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


def _parse_filemap(path: pathlib.Path, exts: set[str]) -> dict[str, pathlib.Path]:
    """Convert a file or directory path into a ``{identifier: audio_path}`` dict.

    For a single file the key is the stem.  For a directory the keys are
    relative-to-root paths (without extension), preserving any subdirectory
    structure.
    """
    if path.is_file():
        return {path.stem: path}
    if path.is_dir():
        files = [
            f for f in sorted(path.rglob("*"))
            if f.is_file() and f.suffix.lower() in exts
        ]
        filemap = {
            f.relative_to(path).with_suffix("").as_posix(): f
            for f in files
        }
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    raise ValueError(f"Invalid path: {path}")


def shared_options(func):
    options = [
        click.option(
            "--model", "-m", required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            help="Path to model checkpoint.",
        ),
        click.option(
            "--input-formats", default="wav,flac,opus,mp3,aac,ogg",
            show_default=True, callback=_validate_exts,
            help="Comma-separated audio file extensions to scan for (directory mode).",
        ),
        click.option(
            "--output-dir", "-o",
            type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
            default=None,
            help="Directory to save output files.  Defaults to the input directory.",
        ),
        click.option(
            "--language", "-l", default=None,
            help="Default language.  Its G2P converters activate; its prefix "
                 "is omitted from output labels.",
        ),
        click.option(
            "--extended-language", "-L",
            default=None, callback=csv_set(str),
            help="Comma-separated additional G2P language tags.",
        ),
        click.option(
            "--g2p",
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            default=None,
            help="Custom G2P pipeline config YAML (overrides inference.g2p from config).",
        ),
        click.option(
            "--oov-handling", default="skip",
            type=click.Choice(["raise", "skip", "force"]), show_default=True,
            help="How to handle OOV phonemes: raise (error), skip (discard sample), force (drop OOV paths).",
        ),
        click.option(
            "--diagnosis", is_flag=True,
            help="Save per-sample diagnosis JSON to <output-dir>/diagnosis.json.",
        ),
        click.option(
            "--plot", is_flag=True,
            help="Save per-sample similarity plots next to TextGrid output.",
        ),
        click.option(
            "--batch-size", type=int, default=8, show_default=True,
            help="Batch size for inference.",
        ),
        click.option(
            "--num-workers", type=int, default=2, show_default=True,
            help="Number of dataloader worker processes.",
        ),
        click.option(
            "--precision", default="32-true", show_default=True,
            help="Precision for inference.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def _run_inference(
    scope: int,
    path: pathlib.Path,
    model: pathlib.Path,
    input_formats: set[str],
    output_dir: pathlib.Path | None,
    language: str | None,
    extended_language: set[str] | None,
    g2p: pathlib.Path | None,
    batch_size: int,
    num_workers: int,
    precision: str,
    topk: int,
    oov_handling: str,
    diagnosis: bool = False,
    plot: bool = False,
):
    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import load_g2p_config, load_inference_model, run_inference
    from inference.data import AudioTextDataset
    from inference.callbacks import DiagnosisCallback, SavePlotCallback, SaveTextGridCallback

    g2p_languages = {language} if language else set()
    if extended_language:
        g2p_languages |= extended_language

    filemap = _parse_filemap(path, input_formats)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    backend, vocabulary, g2p_config = load_inference_model(
        model,
        scope=scope,
        topk=topk,
    )

    if g2p is not None:
        g2p_config = load_g2p_config(g2p, scope=scope)
        g2p_root = g2p.parent if g2p.resolve().is_relative_to(model.parent.resolve()) else ""
    elif g2p_config is not None:
        g2p_root = model.parent
    else:
        raise click.UsageError(
            "The model carries no g2p config. Provide one with --g2p."
        )

    dataset = AudioTextDataset(
        filemap=filemap,
        g2p_config=g2p_config,
        g2p_root=g2p_root,
        vocabulary=vocabulary,
        audio_sample_rate=backend.sample_rate,
        language=g2p_languages if g2p_languages else None,
        oov_handling=oov_handling,
    )

    callbacks = [
        SaveTextGridCallback(
            output_dir=output_dir,
            language=language,
        ),
    ]

    if plot:
        callbacks.append(SavePlotCallback(output_dir=output_dir))

    if diagnosis:
        callbacks.append(DiagnosisCallback(save_path=output_dir / "diagnosis.json"))

    run_inference(
        backend=backend,
        dataset=dataset,
        callbacks=callbacks,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        mode="predict",
    )
    logging.success("Inference completed.", callback=rank_zero_info)


@click.group(cls=DefaultGroup, help="Run forced alignment inference.")
def main():
    pass


@main.default_command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
)
@shared_options
@click.option(
    "--topk", default=10, type=int, show_default=True,
    help="Number of top cosine-similarity frames per token for scoring.",
)
def supervised(**kwargs):
    """Supervised forced alignment inference."""
    _run_inference(ConfigurationScope.FA, **kwargs)


@main.command(name="ssl")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
)
@shared_options
def ssl(**kwargs):
    """Self-supervised forced alignment inference."""
    _run_inference(ConfigurationScope.FA_SSL, **kwargs)


if __name__ == "__main__":
    main()
