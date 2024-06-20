# Script to extract the encodings for all genes from a cellPLM model and save them to a csv file.
# Paper can be found at https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.full
# Github https://github.com/OmicsML/CellPLM
# Directs us to the dropBox folder for downloading the model check-point:
# https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&e=1&dl=0


import json

import click
import pandas as pd
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


def find_gene_embedding_layer(model_configs, model):
    gene_list = model_configs["gene_list"]
    total_number_of_genes = len(gene_list)
    for layer_name in model["model_state_dict"].keys():
        layer_shape = model["model_state_dict"][layer_name].shape
        if (total_number_of_genes in layer_shape) & (len(layer_shape) == 2):
            if layer_shape[1] == 1024:
                return layer_name


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

    embedding_layer = find_gene_embedding_layer(model_configs, model)
    # load the tensor
    encodings = model["model_state_dict"][embedding_layer].numpy()
    # convert to df
    encodings_df = pd.DataFrame(encodings, index=model_configs["gene_list"])

    output_file_name = f"{output_file_dir}/20230926_85M_encoding.csv"
    encodings_df.to_csv(output_file_name)


if __name__ == "__main__":
    main()
