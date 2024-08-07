import io
import warnings
import zipfile
from dataclasses import dataclass

import click
import pandas as pd
import requests

from gene_benchmark.task_retrieval import (
    list_form_to_onehot_form,
    verify_source_of_data,
)
from gene_benchmark.tasks import dump_task_definitions

HIERARCHIES_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
BOTTOM_PATHWAY_TO_GENE_LIST = (
    "https://reactome.org/download/current/ReactomePathways.gmt.zip"
)
PATHWAY_DES = "https://reactome.org/download/current/ReactomePathways.txt"


def get_pathway_to_genes_dict(url):
    gene_list_file = pd.read_csv(url, on_bad_lines="skip", header=None, delimiter="\t")
    pathway_to_gene_df = pd.DataFrame(
        columns=["pathway_description", "pathway", "genes"]
    )
    pathway_to_gene_df[["pathway_description", "pathway"]] = gene_list_file.iloc[:, :2]
    pathway_to_gene_df["genes"] = gene_list_file.iloc[:, 2:].apply(
        lambda x: [str(v) for v in set(x) if not pd.isna(v)], axis=1
    )
    pathway_to_gene_df = pathway_to_gene_df.set_index("pathway")
    return pathway_to_gene_df["genes"].to_dict()


def read_gmt_from_url(url, gmt_filename):
    """
    Downloads a ZIP file from the given URL, extracts the specified GMT file,
    and returns its contents as a dictionary.

    Args:
    ----
        url (str): URL to the ZIP file.
        gmt_filename (str): The name of the GMT file within the ZIP archive.

    Returns:
    -------
        dict: A dictionary where keys are gene set names and values are lists of genes.

    """
    # Step 1: Download the ZIP file
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download ZIP file: {response.status_code}")

    # Step 2: Extract the GMT file from the ZIP archive
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        if gmt_filename not in zip_ref.namelist():
            raise Exception(f"{gmt_filename} not found in the ZIP archive")
        with zip_ref.open(gmt_filename) as gmt_file:
            gmt_content = gmt_file.read().decode("utf-8")

    # Step 3: Parse the GMT file content
    gmt_dict = {}
    for line in gmt_content.strip().split("\n"):
        parts = line.strip().split("\t")
        description = parts[1]
        genes = parts[2:]
        gmt_dict[description] = genes

    return gmt_dict


def get_hierarchy_data(url):
    hierarchy_df = pd.read_csv(
        url, header=None, delimiter="\t", names=["parent", "child"]
    )
    hierarchy_df = hierarchy_df.set_index("parent")

    hierarchies = (
        hierarchy_df.groupby(hierarchy_df.index)["child"].apply(list).to_dict()
    )
    return hierarchies, hierarchy_df


def pathways_2_one_hot(pathways, path_2_gene):
    task_df = pd.DataFrame(columns=["genes"], index=pathways)
    task_df["genes"] = [";".join(path_2_gene.get_genes(path)) for path in task_df.index]
    return list_form_to_onehot_form(
        task_df, participant_col_name="genes", delimiter=";"
    )


def pathway_set_to_task(
    pathways, path_2_gene, pathway_des, pathway_identifier, main_task_directory, verbose
):
    outcomes = pathways_2_one_hot(pathways, path_2_gene)
    outcomes.rename(pathway_des["name"].to_dict())
    symbols = pd.Series(outcomes.index, name="symbol")
    dump_task_definitions(symbols, outcomes, main_task_directory, pathway_identifier)
    if verbose:
        print(
            f"{pathway_identifier}, was created at {main_task_directory} shaped {outcomes.shape}"
        )


def get_gene_descriptions(url):
    return pd.read_csv(
        url,
        on_bad_lines="skip",
        header=None,
        delimiter="\t",
        index_col=0,
        names=["name", "species"],
    )


def get_top_level_dict(hierarchy_df, pathway_des):
    child_set = set(hierarchy_df["child"].values)
    top_level = list(filter(lambda x: not x in child_set, hierarchy_df.index))
    pathway_des["top_level"] = [path in top_level for path in pathway_des.index]
    pathway_des["idx"] = pathway_des.index
    return (
        pathway_des.loc[pathway_des["top_level"], :]
        .groupby(["species"])["idx"]
        .apply(list)
        .to_dict()
    )


@dataclass
class PathwaySeeks:
    pathway_to_gene: dict[str, list[str]]
    hierarchies: dict[str, list[str]]

    def get_genes(self, pathway: list[str], verbose=False) -> list[str]:
        sub_genes = set()
        if pathway in self.pathway_to_gene:
            return self.pathway_to_gene[pathway]
        elif not pathway in self.hierarchies:
            if verbose:
                warnings.warn(
                    f"Pathway {pathway} has no sub pathways and no genes defined"
                )
        else:
            for sub_pathways in self.hierarchies[pathway]:
                sub_genes.update(self.get_genes(sub_pathways))
        self.pathway_to_gene[pathway] = list(sub_genes)
        return list(sub_genes)


@click.command()
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--allow-downloads",
    type=click.BOOL,
    help="If false data files will be downloaded directly from HGNC and reactome, set to true only if you trust the urls above",
    default=False,
)
@click.option(
    "--pathway-identifier",
    "-p",
    type=click.STRING,
    help="Pathway identifier from which we want to create multilabel task.",
    required=True,
    multiple=True,
)
@click.option(
    "--hierarchy-file",
    type=click.STRING,
    help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
    default=None,
)
@click.option(
    "--pathway-description",
    type=click.STRING,
    help="The location of the ReactomePathways.txt file available at https://reactome.org/download-data",
    default=None,
)
@click.option(
    "--pathway-to-gene-file",
    "-m",
    type=click.STRING,
    help="A file with the bottom pathways and the genes involved in the pathways available at https://reactome.org/download-data",
    default=None,
)
@click.option(
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
@click.option(
    "--add-top-level",
    type=click.BOOL,
    default=True,
)
def main(
    main_task_directory,
    allow_downloads,
    pathway_identifier,
    hierarchy_file,
    pathway_description,
    pathway_to_gene_file,
    verbose,
    add_top_level,
):

    hierarchies_file_url = verify_source_of_data(
        hierarchy_file, url=HIERARCHIES_URL, allow_downloads=allow_downloads
    )

    pathway_to_gene_url = verify_source_of_data(
        pathway_to_gene_file,
        url=BOTTOM_PATHWAY_TO_GENE_LIST,
        allow_downloads=allow_downloads,
    )
    pathway_des_url = verify_source_of_data(
        pathway_description, url=PATHWAY_DES, allow_downloads=allow_downloads
    )
    pathway_des = get_gene_descriptions(pathway_des_url)
    hire_dict, hierarchy_df = get_hierarchy_data(hierarchies_file_url)
    path_2_gene = PathwaySeeks(
        read_gmt_from_url(pathway_to_gene_url, "ReactomePathways.gmt"), hire_dict
    )

    for pathway_idx in pathway_identifier:
        pathways = hire_dict[pathway_idx]
        path_name = pathway_des.loc[pathway_identifier, "name"]
        pathway_set_to_task(
            pathways, path_2_gene, pathway_des, path_name, main_task_directory, verbose
        )
    if add_top_level:
        top_dict = get_top_level_dict(hierarchy_df, pathway_des)
        for species, pathways in top_dict.items():
            pathway_set_to_task(
                pathways,
                path_2_gene,
                pathway_des,
                f"top level pathways {species}",
                main_task_directory,
                verbose,
            )


if __name__ == "__main__":
    main()
