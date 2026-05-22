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
    from preprocessing.api import load_config_for_binarization
    from preprocessing.phonemes_binarizer import PhonemesBinarizer

    config_obj = load_config_for_binarization(config, overrides=override)
    binarizer = PhonemesBinarizer(config_obj, eval_mode=eval_mode)

    if vocab_only:
        metadata_list = binarizer.collect_metadata()
        logging.info(f"Collected {len(metadata_list)} metadata items.")
        binarizer.build_vocabulary(metadata_list)
        logging.success("Vocabulary built and plot saved. Exiting.")
        return

    if eval_mode:
        logging.debug("Using evaluation mode.")
    from preprocessing.api import binarize_datasets
    binarize_datasets(binarizer=binarizer)


if __name__ == "__main__":
    main()
