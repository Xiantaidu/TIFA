import pathlib

import click

from lib import logging


@click.command(help="Binarize raw phone datasets.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=click.STRING, required=False,
    help="Override configuration values in dotlist format."
)
@click.option(
    "--vocab", "vocab_only", is_flag=True, default=False, show_default=True,
    help="Build vocabulary from metadata, save a distribution plot, and exit."
)
@click.option(
    "--eval", "eval_mode", is_flag=True, default=False, show_default=True,
    help="Evaluation mode: the whole dataset will be processed as validation set."
)
def main(config: pathlib.Path, override: list[str], vocab_only: bool, eval_mode: bool):
    from preprocessing.api import (
        load_config_for_binarization, build_vocab_from_datasets, binarize_datasets
    )
    from preprocessing.phonemes_binarizer import PhonemeTimingBinarizer
    from preprocessing.texts_binarizer import TextOnlyBinarizer

    config_obj = load_config_for_binarization(config, overrides=override)

    binarizer_classes = [PhonemeTimingBinarizer]
    if config_obj.text_only_data_dir is not None:
        binarizer_classes.append(TextOnlyBinarizer)

    if vocab_only:
        build_vocab_from_datasets(
            config=config_obj,
            binarizer_classes=binarizer_classes,
        )
        return

    if eval_mode:
        logging.debug("Using evaluation mode.")

    binarize_datasets(
        config=config_obj,
        binarizer_classes=binarizer_classes,
        eval_mode=eval_mode
    )


if __name__ == "__main__":
    main()
