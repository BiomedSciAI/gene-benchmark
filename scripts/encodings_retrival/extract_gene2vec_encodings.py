# Script to extract the encodings for all genes from gene2vec and save them to a csv file.
# The encodings are taken from https://github.com/jingcheng-du/Gene2vec/tree/master/pre_trained_emb
import tempfile
from pathlib import Path

import click

GENE2VEC_URL = "https://github.com/jingcheng-du/Gene2vec/blob/6236e0b21fbc367bf8ff5695ed0cc1443861bd1e/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"


def load_gene2vec_encodings():
    pass


def save_encodings():
    pass


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
    help="The path to the directory with the data files",
    default=None,
)
@click.option(
    "--output-file-dir",
    type=click.STRING,
    help="output files path",
)
def main(allow_downloads, input_file_dir, output_file_dir):
    if allow_downloads:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            encodings = load_gene2vec_encodings(model_dir=tmpdir)
    else:
        encodings = load_gene2vec_encodings(model_dir=input_file_dir)

    save_encodings(encodings, output_file_dir)
