import pickle
import tempfile
from pathlib import Path

import click
import pandas as pd
import requests
import torch

GENEFORMER_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/pytorch_model.bin?download=true"
TOKEN_DICT_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary.pkl?download=true"


def download_geneformer(dir_path):
    local_filename = dir_path / "pytorch_model.bin"
    with requests.get(GENEFORMER_URL, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def download_token_dict(dir_path):
    local_filename = dir_path / "token_dictionary.pkl"
    with requests.get(TOKEN_DICT_URL, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def add_tokens_as_index(encodings, token_dict):
    encodings_with_symbols = pd.DataFrame(
        encodings["bert.embeddings.word_embeddings.weight"].detach(),
        index=list(token_dict.keys()),
    )
    encodings_with_symbols = encodings_with_symbols.reset_index()
    encodings_with_symbols = encodings_with_symbols.rename(
        columns={"index": "ensembl_gene_id"}
    )
    return encodings_with_symbols


def save_encodings(encodings, output_file_dir):
    """
    Save the gene encodings to the output dir.

    Args:
    ----
        encodings (pd.DataFrame): the encodings
        output_file_dir (str): path to dir for saving the file

    """
    model_encodings_dir = Path(output_file_dir) / "Geneformer"
    model_encodings_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = model_encodings_dir / "encodings.csv"
    encodings.to_csv(output_file_path, index_label="symbol")


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
            download_token_dict(tmpdir)
            path_to_data = Path(tmpdir) / "pytorch_model.bin"
            path_to_tokens = Path(tmpdir) / "token_dictionary.pkl"
            with open(path_to_tokens, "rb") as file:
                token_dict = pickle.load(file)
            encodings = torch.load(path_to_data, map_location=torch.device("cpu"))
    else:
        path_to_data = Path(input_file_dir) / "pytorch_model.bin"
        encodings = torch.load(path_to_data, map_location=torch.device("cpu"))
        path_to_tokens = Path(input_file_dir) / "token_dictionary.pkl"
        with open(path_to_tokens, "rb") as file:
            token_dict = pickle.load(file)

    encodings = add_tokens_as_index(encodings, token_dict)
    save_encodings(encodings, output_file_dir)


if __name__ == "__main__":
    main()
