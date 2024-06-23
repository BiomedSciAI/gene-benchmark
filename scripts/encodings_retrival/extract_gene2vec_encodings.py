# Script to extract the encodings for all genes from gene2vec and save them to a csv file.
# The encodings are taken from https://github.com/jingcheng-du/Gene2vec/tree/master/pre_trained_emb
from pathlib import Path

import click
import pandas as pd

GENE2VEC_URL = "https://github.com/jingcheng-du/Gene2vec/blob/6236e0b21fbc367bf8ff5695ed0cc1443861bd1e/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"


def load_encodings_w2v_format(path_to_data):
    return pd.read_csv(
        path_to_data,
        sep=" ",
        index_col=0,
        header=None,
        skiprows=1,
    )


def save_encodings(encodings, output_file_dir):
    """
    Save the gene encodings to the output dir.

    Args:
    ----
        encodings (pd.DataFrame): the encodings
        model_type (str): "blood" or "human"
        output_file_dir (str): path to dir for saving the file

    """
    model_encodings_dir = Path(output_file_dir) / "Gene2Vec"
    model_encodings_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = model_encodings_dir / "encodings.csv"
    encodings.to_csv(output_file_path, index_label="symbol")


@click.command()
@click.option(
    "--allow-downloads",
    "-l",
    type=click.BOOL,
    help=f"download files directly from {GENE2VEC_URL}, use this option only if you trust the URL.",
    default=False,
)
@click.option(
    "--input-file-dir",
    type=click.STRING,
    help="The path to the csv file with the encodings",
    default=None,
)
@click.option(
    "--output-file-dir",
    type=click.STRING,
    help="output files path",
)
def main(allow_downloads, input_file_dir, output_file_dir):

    if allow_downloads:
        encodings = load_encodings_w2v_format(GENE2VEC_URL)
    else:
        encodings = load_encodings_w2v_format(input_file_dir)

    save_encodings(encodings, output_file_dir)
