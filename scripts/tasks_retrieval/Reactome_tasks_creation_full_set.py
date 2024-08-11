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


def get_pathway_symbols_dict(
    url: str, gmt_filename: str, allow_downloads: bool
) -> dict[str, list[str]]:
    """
    Downloads a ZIP file from the given URL, extracts the specified GMT file,
    and returns its contents as a dictionary.

    Args:
    ----
        url (str): URL of path for a zip file
        gmt_filename (str): The name of the GMT file within the ZIP archive.
        allow_downloads (bool): indicator if the file is a url or not

    Raises:
    ------
        Exception: _description_

    Returns:
    -------
        dict[str,list[str]]: A dictionary where keys are pathway names and values are lists of genes.

    """
    if allow_downloads:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download ZIP file: {response.status_code}")
        fstream = io.BytesIO(response.content)
    else:
        fstream = url
    with zipfile.ZipFile(fstream) as zip_ref:
        gmt_content = get_gmt_from_zip(zip_ref, gmt_filename)
    return gmt_file_to_dictionary(gmt_content)


def get_gmt_from_zip(zip_ref, gmt_filename: str):
    """
    extract the GMT file from a zip archive.

    Args:
    ----
        zip_ref (zipfile.ZipFile): Zip archive containing a gmt file
        gmt_filename (str): The archived GMT file name

    Raises:
    ------
        Exception: raised if the zip archive doesn't contain the file

    Returns:
    -------
        str: the gmt_file file content

    """
    if gmt_filename not in zip_ref.namelist():
        raise Exception(f"{gmt_filename} not found in the ZIP archive")
    with zip_ref.open(gmt_filename) as gmt_file:
        gmt_content = gmt_file.read().decode("utf-8")
    return gmt_content


def gmt_file_to_dictionary(gmt_content: str) -> dict[str, list[str]]:
    r"""
    Turns a reactom gmt file into a dictionary.

    Args:
    ----
        gmt_content (str): A file where each line is separated by \n and values by \t

    Returns:
    -------
        dict[str,list[str]]: A dictionary where keys are pathway names and values are lists of genes.

    """
    gmt_dict = {}
    for line in gmt_content.strip().split("\n"):
        parts = line.strip().split("\t")
        description = parts[1]
        genes = parts[2:]
        gmt_dict[description] = genes
    return gmt_dict


def get_hierarchy_data(url: str) -> [dict[str, str], pd.DataFrame]:
    """
    loads and returns the pathways hierarchies.

    Args:
    ----
        url (str): the location of the file

    Returns:
    -------
        [dict[str,str],pd.DataFrame]: a tuple with dictionary with parent pathways as keys and children as values and in a data frame format

    """
    hierarchy_df = pd.read_csv(
        url, header=None, delimiter="\t", names=["parent", "child"]
    )
    hierarchy_df = hierarchy_df.set_index("parent")

    hierarchies = (
        hierarchy_df.groupby(hierarchy_df.index)["child"].apply(list).to_dict()
    )
    return hierarchies, hierarchy_df


def pathways_2_one_hot(pathways: list[str], path_2_gene) -> pd.DataFrame:
    """
    create a one hot data frame with the genes in the rows and pathways as columns.

    Args:
    ----
        pathways (list[str]): A list of pathways to analyze
        path_2_gene (PathwaySeeks): PathwaySeeks object with the corresponding gene to list of genes

    Returns:
    -------
        pd.DataFrame: a one hot data frame

    """
    task_df = pd.DataFrame(columns=["genes"], index=pathways)
    task_df["genes"] = [";".join(path_2_gene.get_genes(path)) for path in task_df.index]
    return list_form_to_onehot_form(
        task_df, participant_col_name="genes", delimiter=";"
    )


def pathway_set_to_task(
    pathways: list[str],
    path_2_gene,
    pathway_des: pd.DataFrame,
    pathway_identifier: str,
    main_task_directory: str,
    verbose: bool,
):
    """
    create a task from a set of pathways.

    Args:
    ----
        pathways (list[str]): the list of pathways to be collected into the task
        path_2_gene (PathwaySeeks): PathwaySeeks object with the corresponding gene to list of genes
        pathway_des (pd.DataFrame): Dataframe with pathways identifier as index and a name column
        pathway_identifier (str): the pathways identifier
        verbose (bool): if true will do some prints

    """
    outcomes = pathways_2_one_hot(pathways, path_2_gene)
    outcomes.rename(pathway_des["name"].to_dict())
    symbols = pd.Series(outcomes.index, name="symbol")
    dump_task_definitions(symbols, outcomes, main_task_directory, pathway_identifier)
    if verbose:
        print(
            f"{pathway_identifier}, was created at {main_task_directory} shaped {outcomes.shape}"
        )


def get_gene_descriptions(url: str) -> pd.DataFrame:
    """
    retrieves the data frame description.

    Args:
    ----
        url (str): the file location

    Returns:
    -------
        pd.DataFrame: data frame with pathways descriptions

    """
    return pd.read_csv(
        url,
        on_bad_lines="skip",
        header=None,
        delimiter="\t",
        index_col=0,
        names=["name", "species"],
    )


def get_top_level_dict(
    hierarchy_df: pd.DataFrame, pathway_des: pd.DataFrame
) -> dict[str, list[str]]:
    """
    returns a dictionary with the top pathways of each species.

    Args:
    ----
        hierarchy_df (pd.DataFrame): A data frame containing the hierarchy of pathways
        pathway_des (pd.DataFrame): A data frame containing the

    Returns:
    -------
        dict[str,list[str]]: _description_

    """
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
    """
    The data from reactom contains the gene list for bottom level pathways only to prevent redundant calculations
    this class saves the gene list of each pathway when it encounter its.
    """

    pathway_to_gene: dict[str, list[str]]
    hierarchies: dict[str, list[str]]

    def get_genes(self, pathway: list[str], verbose=False) -> list[str]:
        """
        retrives the gene list of a pathway by merging all of it's descendants genes.

        Args:
        ----
            pathway (list[str]): A list of pathways
            verbose (bool, optional): If true warnings will be printed. Defaults to False.

        Returns:
        -------
            list[str]: a list of gene symbols

        """
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
    multiple=True,
    default=None,
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
    "--add-top-levels",
    help="If true will create a task per all of the top level human pathways and per each top level pathway individually for Homo sapiens",
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
    add_top_levels,
):
    urls_dict = {}
    for nme, file, url in zip(
        ["hierarchies", "pathway_to_gene", "pathway_to_des"],
        [hierarchy_file, pathway_to_gene_file, pathway_description],
        [HIERARCHIES_URL, BOTTOM_PATHWAY_TO_GENE_LIST, PATHWAY_DES],
    ):
        urls_dict[nme] = verify_source_of_data(
            file, url=url, allow_downloads=allow_downloads
        )

    pathway_des = get_gene_descriptions(urls_dict["pathway_to_des"])
    hire_dict, hierarchy_df = get_hierarchy_data(urls_dict["hierarchies"])
    path_2_gene = PathwaySeeks(
        get_pathway_symbols_dict(urls_dict["pathway_to_gene"], "ReactomePathways.gmt"),
        hire_dict,
    )
    if add_top_levels:
        top_dict = get_top_level_dict(hierarchy_df, pathway_des)
        pathway_set_to_task(
            top_dict["Homo sapiens"],
            path_2_gene,
            pathway_des,
            "Pathways top level homo sapiens",
            main_task_directory,
            verbose,
        )
        pathway_identifier = pathway_identifier + tuple(top_dict["Homo sapiens"])

    for pathway_idx in pathway_identifier:
        pathways = hire_dict[pathway_idx]
        path_name = f"pathway {pathway_des.loc[pathway_idx, 'name']}"
        pathway_set_to_task(
            pathways, path_2_gene, pathway_des, path_name, main_task_directory, verbose
        )


if __name__ == "__main__":
    main()
