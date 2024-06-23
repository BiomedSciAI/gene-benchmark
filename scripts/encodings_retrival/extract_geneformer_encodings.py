import tempfile
from pathlib import Path

import click
import torch

GENEFORMER_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/pytorch_model.bin?download=true"


def download_geneformer(GENEFORMER_URL):
    pass


def save_encodings():
    pass


@click.command()
@click.option(
    "--allow-downloads",
    "-l",
    type=click.BOOL,
    help=f"download files directly from {GENEFORMER_URL}, use this option only if you trust the URL.",
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
    default="./encodings",
)
def main(allow_downloads, input_file_dir, output_file_dir):

    if allow_downloads:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            download_geneformer(tmpdir)
            encodings = torch.load(tmpdir, map_location=torch.device("cpu"))
    else:
        path_to_data = Path(input_file_dir) / "pytorch_model.bin"
        encodings = torch.load(path_to_data, map_location=torch.device("cpu"))

    save_encodings(encodings, output_file_dir)


if __name__ == "__main__":
    main()
