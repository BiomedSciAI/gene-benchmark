import click
import pandas as pd
import requests

from gene_benchmark.task_retrieval import (
    list_form_to_onehot_form,
    verify_source_of_data,
)
from gene_benchmark.tasks import dump_task_definitions

TOP_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"


def get_token_link_for_symbols(symbols: list[str]) -> str:
    """
    Creates an analysis service pathways link for a given symbol list.

    Args:
    ----
        symbols (list[str]): list of symbols to create a pathways data file for

    Returns:
    -------
        str: the to the csv file with the pathways for the symbols

    """
    token = get_token(symbols)
    return f"https://reactome.org/AnalysisService/download/{token}/pathways/TOTAL/result.csv"


def get_symbol_list(
    url: str = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json",
) -> list[str]:
    """
    Retrieves the symbol list from a HGNC json like file.

    Args:
    ----
        url (str, optional): url for the json file download. Defaults to "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json".

    Returns:
    -------
        list[str]: list of symbols

    """
    with requests.get(url) as response:
        response.raise_for_status()
        reactome_res = response.json()
    return [v["symbol"] for v in reactome_res["response"]["docs"]]


def get_token(
    identifiers: list[str],
    projection_url: str = "https://reactome.org/AnalysisService/identifiers/projection",
) -> str:
    """
    Data retrieval from Reactome API requires the use of token that represent a list of identifiers,
       the method use the  AnalysisService API to get the token for a given identifiers list.

    Args:
    ----
        identifiers (list[str]): List of identifiers
        projection_url (str, optional): Analysis service link. Defaults to "https://reactome.org/AnalysisService/identifiers/projection".

    Returns:
    -------
        str: A Reactome Analysis service token

    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "text/plain",
    }
    symbols = "\n".join(identifiers)
    response = requests.post(
        projection_url,
        headers=headers,
        data=symbols,
    )
    return response.json()["summary"]["token"]


def get_top_level_pathway(hierarchies_df: pd.DataFrame) -> set[str]:
    """
    Returns the top level pathways from the table of pathways hierarchies.
        top level are defined as pathways without a parent.

    Args:
    ----
        hierarchies_df (pd.DataFrame): A data frame with a parent and child headers

    Returns:
    -------
        set[str]: a set of top level pathways

    """
    pathway_that_are_parents = set(hierarchies_df["parent"].values)
    pathway_that_are_children = set(hierarchies_df["child"].values)
    pathway_who_are_just_parents = pathway_that_are_parents - pathway_that_are_children
    return pathway_who_are_just_parents


def create_top_level_task(
    hierarchies_df: pd.DataFrame,
    df_path: pd.DataFrame,
    entities_name: str = "symbol",
    pathway_names: str = "Pathway name",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Creates a top level tasks.

    Args:
    ----
        hierarchies_df (pd.DataFrame): The pathways hierarchies table used to find the top pathways
        df_path (pd.DataFrame): The pathways themselves, used to extract the gene list.
        entities_name (str, optional): name of the entities. Defaults to 'symbol'.
        pathway_names (str, optional): names of the pathways (converted from identifiers). Defaults to "Pathway name".

    Returns:
    -------
        tuple[pd.Series,pd.DataFrame]: _description_

    """
    top_level = get_top_level_pathway(hierarchies_df)
    top_in_file_paths = top_level.intersection(set(df_path.index))
    df_path_top = df_path.loc[list(top_in_file_paths), :]
    df_path_top.index = df_path_top[pathway_names]
    outcomes = list_form_to_onehot_form(df_path_top)
    symbols = pd.Series(outcomes.index, name=entities_name)
    return symbols, outcomes


@click.command()
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
    help="name for the specific task",
    default="Pathways HGNC",
)
@click.option(
    "--allow-downloads",
    type=click.BOOL,
    help="If false data files will be downloaded directly from HGNC and reactome, set to true only if you trust the urls above",
    default=False,
)
@click.option(
    "--pathways-file",
    type=click.STRING,
    help="Path to the pathways files from reactome available using the analysis GUI",
    default=None,
)
@click.option(
    "--pathways-relation-file",
    type=click.STRING,
    help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
    default=None,
)
@click.option(
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
def main(
    main_task_directory,
    task_name,
    allow_downloads,
    pathways_file,
    pathways_relation_file,
    verbose,
):

    reactom_url = (
        get_token_link_for_symbols(get_symbol_list()) if allow_downloads else ""
    )

    pathways_file = verify_source_of_data(
        pathways_file, url=reactom_url, allow_downloads=allow_downloads
    )
    pathways_relation_file = verify_source_of_data(
        pathways_relation_file, url=TOP_PATHWAYS_URL, allow_downloads=allow_downloads
    )
    df_path = pd.read_csv(pathways_file, index_col="Pathway identifier")

    hierarchies_df = pd.read_csv(
        pathways_relation_file, delimiter="\t", header=0, names=["parent", "child"]
    )
    symbols, outcomes = create_top_level_task(hierarchies_df, df_path)
    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)
    if verbose:
        print(
            f"{task_name} was created at {main_task_directory} shaped {outcomes.shape}"
        )
    return


if __name__ == "__main__":
    main()
