import re
from pathlib import Path

import click
import pandas as pd
import requests
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from gene_benchmark.descriptor import NCBIDescriptor

GENE_SYMBOL_URL = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json"


def get_symbol_list(url=GENE_SYMBOL_URL):
    with requests.get(url) as response:
        response.raise_for_status()
        reactome_res = response.json()
    return [v["symbol"] for v in reactome_res["response"]["docs"]]


def get_descriptions(gene_symbols: list, verbose):
    prompts_maker = NCBIDescriptor()
    prompts = prompts_maker.describe(entities=pd.Series(gene_symbols))
    prompts.index = gene_symbols
    prompts = prompts.dropna()
    if verbose:
        print("Created gene descriptions")
        all_text = " ".join(prompts.values)
        words = re.findall(r"\b\w+\b", all_text.lower())
        print(f"Total number of words in vocabulary: {len(set(words))}")
    return prompts


def create_bag_of_words(corpus: pd.Series):
    vectorizer = CountVectorizer(max_features=1024)
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), index=corpus.index)


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
@click.option("--verbose", "-v", is_flag=True, default=True)
def main(allow_downloads: bool, input_file: str, output_file_dir: str, verbose: bool):

    if allow_downloads:
        gene_list = get_symbol_list(GENE_SYMBOL_URL)
        if verbose:
            print(f"Downloaded gene list from {GENE_SYMBOL_URL}")
    else:
        with open(input_file) as file:
            gene_list = yaml.safe_load(file)
        if verbose:
            print(f"Loaded gene list from {input_file}")

    prompts = get_descriptions(gene_list, verbose)
    encodings = create_bag_of_words(corpus=prompts)
    save_encodings(encodings, output_file_dir)


if __name__ == "__main__":
    main()
