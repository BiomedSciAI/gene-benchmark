import re
from pathlib import Path

import click
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from gene_benchmark.descriptor import NCBIDescriptor
from scripts.tasks_retrieval.task_retrieval import GENE_SYMBOL_URL, get_symbol_list


def get_descriptions(gene_symbols: list):
    """
    Get gene descriptions from NCBI, the genes with no description will be dropped.

    Args:
    ----
        gene_symbols (list): a list of gene symbol names

    Returns:
    -------
        A pd.Series with the gene symbols as index and the descriptions as values.

    """
    prompts_maker = NCBIDescriptor()
    prompts = prompts_maker.describe(entities=pd.Series(gene_symbols))
    prompts.index = gene_symbols
    return prompts.dropna()


def create_bag_of_words(corpus: pd.Series, max_features: int = 1024):
    """
    Create bag of words for a given corpus using sklearn 'CountVectorizer'.

    Args:
    ----
        corpus (pd.Series):  A pd.Series with the gene symbols as index and the descriptions as values.
        max_features (int): Number of features to use for the bag of words

    Returns:
    -------
        A pd.DataFrame with the gene symbols as index and the bag of word counts as the columns

    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), index=corpus.index)


def print_number_of_unique_words(series: pd.Series):
    """
    Print number of unique words in series.

    Args:
    ----
        series (pd.Series):  A pd.Series with the gene symbols as index and the descriptions as values.

    """
    all_text = " ".join(series.values)
    words = re.findall(r"\b\w+\b", all_text.lower())
    print(f"Total number of words: {words}")
    print(f"Total number of unique words in vocabulary: {len(set(words))}")


def save_encodings(
    encodings: pd.DataFrame,
    output_file_dir: str,
    sub_dir_name: str = "Bag_of_words",
    file_name: str = "encodings.csv",
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
    help="The path to the yaml file with the gene names",
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
    else:
        with open(input_file) as file:
            gene_list = yaml.safe_load(file)

    if verbose:
        print("Started extracting descriptions")
    prompts = get_descriptions(gene_list)
    if verbose:
        print_number_of_unique_words(prompts)
    encodings = create_bag_of_words(corpus=prompts)
    save_encodings(encodings, output_file_dir)


if __name__ == "__main__":
    main()
