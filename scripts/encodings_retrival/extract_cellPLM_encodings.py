# Script to extract the encodings for all genes from a cellPLM model and save them to a csv file.
# Paper can be found at https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.full
# Github https://github.com/OmicsML/CellPLM
# Directs us to the dropBox folder for downloading the model check-point:
# https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&e=1&dl=0


import json

import click
import torch


def download_files():
    pass


def load_files(input_file_dir):
    model_file = f"{input_file_dir}/20230926_85M.best.ckpt"
    model_config_file = f"{input_file_dir}/20230926_85M.config.json"
    with open(model_config_file) as f:
        model_configs = json.load(f)

    model = torch.load(model_file, map_location=torch.device("cpu"))
    return model_configs, model


@click.command()
@click.option(
    "--allow-downloads",
    "-l",
    type=click.BOOL,
    help="download files directly from urls, from the box, use this option only if you trust the URLs.",
    default=False,
)
@click.option(
    "--input-file-dir",
    type=click.STRING,
    help="The path to the directory with the data files",
    default=None,
)
@click.option(
    "--output-file-dir",
    type=click.STRING,
    help="output files path",
    default="./encodings",
)
def main(allow_downloads, input_file_dir, output_file_dir):

    if allow_downloads:
        model_configs, model = download_files()
    else:
        model_configs, model = load_files(input_file_dir)


if __name__ == "__main__":
    main()
