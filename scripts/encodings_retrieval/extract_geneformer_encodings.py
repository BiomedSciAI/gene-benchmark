import pickle
import sys
import tempfile
from pathlib import Path

import click
import pandas as pd
import requests
import torch

sys.path.append("./scripts/tasks_retrieval")
from task_retrieval import get_symbols

GENEFORMER_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/pytorch_model.bin?download=true"
TOKEN_DICT_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary.pkl?download=true"


def download_file_from_url(url, file_name, dir_path, chunk_size=8192):
    """
    Download a file fom a url path.

    Args:
    ----
        url (str): url of target file
        file_name (str): name of output file
        dir_path (pathlib.Path): path to dir for saving the file
        chunk_size (int): size of data to be streamed

    Returns:
    -------
        The path of the saved file

    """
    local_filename = dir_path / file_name
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
    return local_filename


def add_tokens_as_index(encodings, token_dict):
    """
    Add tokens names to the encodings file.

    Args:
    ----
        encodings (torch.tensor): the encodings
        token_dict (dict): mapping of symbol name to index

    Returns:
    -------
        encodings with the symbols

    """
    encodings_with_symbols = pd.DataFrame(
        encodings["bert.embeddings.word_embeddings.weight"].detach(),
        index=list(token_dict.keys()),
    )
    return encodings_with_symbols


def save_encodings(
    encodings, output_file_dir, sub_dir_name="Geneformer", file_name="encodings.csv"
):
    """
    Save the gene encodings to the output dir.

    Args:
    ----
        encodings (pd.DataFrame): the encodings
        output_file_dir (str): path to dir for saving the file
        sub_dir_name (str): name od sub directory to save the encodings.
        file_name (str): name of encoding file

    """
    model_encodings_dir = Path(output_file_dir) / sub_dir_name
    model_encodings_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = model_encodings_dir / file_name
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
@click.option(
    "--as-symbols",
    type=click.STRING,
    help="If true sets the index to be symbols when duplicates are merged via mean values",
    default=True,
)
def main(allow_downloads, input_file_dir, output_file_dir, as_symbols):

    if allow_downloads:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            path_to_data = download_file_from_url(
                GENEFORMER_URL, file_name="pytorch_model.bin", dir_path=tmpdir
            )
            path_to_tokens = download_file_from_url(
                TOKEN_DICT_URL, file_name="token_dictionary.pkl", dir_path=tmpdir
            )
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
    if as_symbols:
        encodings.index = get_symbols(encodings.index, dropna=False)
        encodings = encodings.loc[~encodings.index.isna(), :]
    save_encodings(encodings, output_file_dir)


if __name__ == "__main__":
    main()
