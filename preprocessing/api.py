import pathlib
from collections.abc import Mapping

from lib import logging
from lib.config.formatter import format_model
from lib.config.io import load_raw_config
from lib.config.schema import BinarizerConfig, RootConfig
from lib.vocabulary import Vocabulary, VocabularyBuilder

__all__ = [
    "load_config_for_binarization",
    "build_vocab_from_datasets",
    "build_shared_vocab",
    "binarize_datasets",
]


def load_config_for_binarization(
        config_path: pathlib.Path,
        scope: int = 0,
        overrides: list[str] = None
) -> BinarizerConfig:
    config = load_raw_config(config_path, inherit=True, overrides=overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    print(format_model(config.binarizer))
    return config.binarizer


def build_shared_vocab(vocab_config, metadata_list) -> tuple[Vocabulary, Mapping]:
    builder = VocabularyBuilder(
        global_symbols=vocab_config.global_symbols,
        stop_symbols=vocab_config.stop_symbols,
        merged_groups=vocab_config.merged_groups,
        peers=vocab_config.peers,
    )
    for item in metadata_list:
        builder.add(item.raw_symbols, default_language=item.language)
    vocab = builder.build()
    counter = builder.counter()
    return vocab, counter


def build_vocab_from_datasets(
        config: BinarizerConfig,
        binarizer_classes: list[type],
):
    binarizers = [cls(config=config) for cls in binarizer_classes]
    all_metadata = []
    for b in binarizers:
        metadata = b.collect_metadata()
        logging.info(
            f"Collected {len(metadata)} metadata items "
            f"from '{b.data_dir.as_posix()}' ({b.__class__.__name__})."
        )
        all_metadata.extend(metadata)
    logging.info(f"Collected {len(all_metadata)} metadata items in total.")
    vocab, counter = build_shared_vocab(config.vocabulary, all_metadata)
    binarizers[0].save_vocab_plot(counter)
    logging.success("Vocabulary built and plot saved.")


def binarize_datasets(
        config: BinarizerConfig,
        binarizer_classes: list[type],
        eval_mode: bool = False
):
    binarizers = [
        binarizer_classes[0](config=config, eval_mode=eval_mode),
        *[
            cls(config=config, eval_mode=eval_mode, aux_mode=True)
            for cls in binarizer_classes[1:]
        ]
    ]

    if len(binarizers) == 1:
        logging.info(f"Starting binarizer: {binarizers[0].__class__.__name__}.")
        binarizers[0].process()
        logging.success("Binarization completed.")
        return

    # Multi-dataset: collect metadata from all
    per_metadata = []
    for b in binarizers:
        metadata = b.collect_metadata()
        logging.info(
            f"Dataset '{b.data_dir.as_posix()}': {len(metadata)} items "
            f"({b.__class__.__name__})."
        )
        per_metadata.append(metadata)
    all_metadata = [item for metadata in per_metadata for item in metadata]
    if not all_metadata:
        raise RuntimeError("No metadata items found in any dataset.")

    # Build shared vocabulary
    shared_vocab, counter = build_shared_vocab(config.vocabulary, all_metadata)
    for b in binarizers:
        b.vocabulary = shared_vocab

    # Main binarizer handles plot
    binarizers[0].save_vocab_plot(counter)

    # Each binarizer builds dataset independently
    for b, metadata in zip(binarizers, per_metadata):
        b.build_dataset(metadata)

    # Each binarizer saves its own auxiliary files
    for b in binarizers:
        b.save_auxiliary_files()

    logging.success("Binarization completed.")
