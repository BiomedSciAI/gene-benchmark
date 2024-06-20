# Script to extract the encodings for all genes from a cellPLM model and save them to a csv file.
# Paper can be found at https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.full
# Github https://github.com/OmicsML/CellPLM
# Directs us to the dropBox folder for downloading the model check-point:
# https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&e=1&dl=0


import json
import tempfile
from pathlib import Path

import click
import pandas as pd
import requests
import torch

MODEL_URLS = {
    "config.json": "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h/ckpt/20230926_85M.config.json?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0",
    "best.ckpt": "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h/ckpt/20230926_85M.best.ckpt?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0",
}


def download_file_from_dropbox(url, name, output_dir):

    modified_url = url.replace("dl=0", "dl=1")
    response = requests.get(modified_url, stream=True)
    output = output_dir / name
    if response.status_code == 200:
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully to {output}")
    else:
        print(f"Failed to download file. HTTP status code: {response.status_code}")


def load_files(input_file_dir):
    """
    Load the config.json file and the best.ckpt file from a directory.

    Args:
    ----
    input_file_dir (str) : path to directory

    Returns:
    -------
    model_configs: the model config file
    model: the model from the ckpt

    """
    model_file = f"{input_file_dir}/best.ckpt"
    model_config_file = f"{input_file_dir}/config.json"
    with open(model_config_file) as f:
        model_configs = json.load(f)

    model = torch.load(model_file, map_location=torch.device("cpu"))
    return model_configs, model


def find_gene_embedding_layer(model_configs, model):
    """
    Find the gene encoding layer from a model state dict.
    The encoding layer has a shape of `[num genes, embedding width]`.
    This method relies on the fact that the encoding size of cellPLM is 1024.

    Args:
    ----
    model_configs: the model config file
    model: the model from the ckpt

    Returns:
    -------
    layer_name: the name of the layer with the gene encodings.

    """
    gene_list = model_configs["gene_list"]
    total_number_of_genes = len(gene_list)
    for layer_name in model["model_state_dict"].keys():
        layer_shape = model["model_state_dict"][layer_name].shape
        if (total_number_of_genes in layer_shape) & (len(layer_shape) == 2):
            if layer_shape[1] == 1024:
                return layer_name


def save_encodings(encodings, output_file_dir):
    """
    Save the gene encodings to the output dir.

    Args:
    ----
        encodings (pd.DataFrame): the encodings
        output_file_dir (str): path to dir for saving the file

    """
    model_encodings_dir = Path(output_file_dir) / "cellPLM"
    model_encodings_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = model_encodings_dir / "encodings.csv"
    encodings.to_csv(output_file_path)


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
    help="The path to the directory with the data files. there needs to be two files: best.ckpt = the model ckpt, config.json = the config file",
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
            for name, url in MODEL_URLS.items():
                download_file_from_dropbox(url, name, output_dir=tmpdir)
            model_configs, model = load_files(tmpdir)
    else:
        model_configs, model = load_files(input_file_dir)

    embedding_layer = find_gene_embedding_layer(model_configs, model)
    encodings = model["model_state_dict"][embedding_layer].numpy()
    encodings_df = pd.DataFrame(encodings, index=model_configs["gene_list"])

    save_encodings(encodings_df, output_file_dir)


if __name__ == "__main__":
    main()
