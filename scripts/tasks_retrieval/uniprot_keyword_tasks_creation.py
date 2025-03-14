import gzip
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import click
import pandas as pd
import requests

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrieval.task_retrieval import GENE_SYMBOL_URL


def get_gene_protein_keyword_dfs(gene_proteins: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Create dict of DataFrames for all UniProt keywords.

    Args:
    ----
        gene_proteins (list[dict]): list of proteins that have gene names in their metadata

    Returns:
    -------
        dict[str, pd.DataFrame]: dict with keys according to keyword category and values
          are DataFrames with gene symbol index, keyword value columns and binary values
          representing whether the gene symbol has the keyword value.

    """
    category_gene_kw_map = create_category_gene_kw_map(gene_proteins)

    return {c: make_gene_kw_df(gkm) for c, gkm in category_gene_kw_map.items()}


def create_category_gene_kw_map(gene_proteins: list[dict]) -> dict[str, dict[str, set]]:
    category_dict = defaultdict(lambda: defaultdict(set))

    for gene_protein in gene_proteins:
        gene_symbol = gene_protein["genes"][0]["geneName"]["value"]

        for kw in gene_protein.get("keywords", []):
            category = kw["category"]
            name = kw["name"]
            category_dict[category][gene_symbol].add(name)
    return category_dict


def make_gene_kw_df(gene_kw_map: dict[str, set]) -> pd.DataFrame:
    all_keywords = sorted({kw for keywords in gene_kw_map.values() for kw in keywords})

    rows = []
    for gene, keywords in gene_kw_map.items():
        row = [1 if kw in keywords else 0 for kw in all_keywords]
        rows.append([gene] + row)

    return pd.DataFrame(rows, columns=["Gene"] + all_keywords).set_index("Gene")


def download_and_load_json_gz(url: str) -> dict:
    """
    Download and gunzip a json from a url.

    Args:
    ----
        url (str): url to download

    Returns:
    -------
        dict: contents of json.gz as a dict

    """
    response = requests.get(url)
    response.raise_for_status()
    compressed_data = BytesIO(response.content)
    with gzip.GzipFile(fileobj=compressed_data) as gz:
        json_data = json.load(gz)

    return json_data


def get_uniprot_human_protein_features(
    file: str | None = None, allow_downloads: bool = False
) -> dict:
    """Get UniProt Human Proteins data from file or from server."""
    if file and Path(file).exists():
        with open(file) as f:
            return json.load(f)
    url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&download=true&format=json&query=%28*%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29"
    if allow_downloads:
        uniprot_features = download_and_load_json_gz(url)
    if file:
        with open(file, "w") as f:
            json.dump(uniprot_features, f)

    return uniprot_features


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
    help="The path to the yaml file with the gene names. If omitted, `allow-downloads` must be True",
    default=None,
)
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--task-name",
    "-n",
    type=click.STRING,
    multiple=True,
    help="Name for the task based on UniProt keyword category. Must be from this list:"
    "  ['Biological process', 'Cellular component', 'Coding sequence diversity', 'Disease', "
    "    'Domain', 'Ligand', 'Molecular function', 'PTM', 'Technical term']"
    "Can be multiply defined. Defaults to creating all of the possible keyword tasks.",
    default=[
        "Biological process",
        "Cellular component",
        "Coding sequence diversity",
        "Disease",
        "Domain",
        "Ligand",
        "Molecular function",
        "PTM",
        "Technical term",
    ],
)
def main(
    allow_downloads: bool, input_file: str, main_task_directory: str, task_name: str
):
    """
    Create gene protein structural domain task.

    This is a multilabel task based on the presence or absence of protein keywords as
    compiled by UniProt.

    This task is only defined for genes that code for proteins. Each protein can have
    multiple keyword values in different locations of its sequence. This task does not count
    copies, but only lists the presence or absence of structural domains on the protein
    products of the gene symbol.
    """
    proteins = get_uniprot_human_protein_features(input_file, allow_downloads)
    # restrict proteins to those with associated named genes
    gene_proteins = [
        i for i in proteins["results"] if "genes" in i and "geneName" in i["genes"][0]
    ]
    gene_keyword_df_dict = get_gene_protein_keyword_dfs(gene_proteins)

    for task in task_name:  # task_name is multiply defined, so a tuple
        dump_task_definitions(
            entities=pd.Series(gene_keyword_df_dict[task].index).rename("symbol"),
            outcomes=gene_keyword_df_dict[task],
            main_task_directory=main_task_directory,
            task_name="UniProt keyword " + task,
        )


if __name__ == "__main__":
    main()
