from pathlib import Path

import click
import pandas as pd
import requests
import yaml

from gene_benchmark.descriptor import NCBIDescriptor

GENE_SYMBOL_URL = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json"


def get_symbol_list(
    url=GENE_SYMBOL_URL,
) -> list[str]:
    with requests.get(url) as response:
        response.raise_for_status()
        reactome_res = response.json()
    return [v["symbol"] for v in reactome_res["response"]["docs"]]


def save_encodings(
    encodings, output_file_dir, sub_dir_name="Bag_of_words", file_name="encodings.csv"
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
    help=f"download files directly from {GENE_SYMBOL_URL}, use this option only if you trust the URL.",
    default=False,
)
@click.option(
    "--input-file",
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
def main(allow_downloads: bool, input_file: str, output_file_dir: str):

    if allow_downloads:
        gene_list = get_symbol_list(GENE_SYMBOL_URL)
    else:
        with open(input_file) as file:
            gene_list = yaml.safe_load(file)

    prompts_maker = NCBIDescriptor()
    prompts = prompts_maker.describe(entities=pd.Series(gene_list))

    save_encodings(prompts, output_file_dir)


if __name__ == "__main__":
    main()
