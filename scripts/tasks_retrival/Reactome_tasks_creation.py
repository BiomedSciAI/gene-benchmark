import click
import pandas as pd
import requests
from task_retrieval import verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrival.task_retrieval import list_form_to_onehot_form

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
    url="https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json",
) -> list[str]:
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
    default="Pathways",
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
    "--top-pathways-file",
    type=click.STRING,
    help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
    default=None,
)
def main(
    main_task_directory, task_name, allow_downloads, pathways_file, top_pathways_file
):

    if allow_downloads:
        reactom_url = (
            get_token_link_for_symbols(get_symbol_list()) if allow_downloads else ""
        )

    pathways_file = verify_source_of_data(
        pathways_file, url=reactom_url, allow_downloads=allow_downloads
    )
    top_pathways_file = verify_source_of_data(
        top_pathways_file, url=TOP_PATHWAYS_URL, allow_downloads=allow_downloads
    )
    df_path = pd.read_csv(pathways_file, index_col="Pathway identifier")

    hierarchies_df = pd.read_csv(
        top_pathways_file, delimiter="\t", header=0, names=["parent", "child"]
    )
    top_level = get_top_level_pathway(hierarchies_df)

    top_in_file_paths = top_level.intersection(set(df_path.index))
    df_path_top = df_path.loc[list(top_in_file_paths), :]
    df_path_top.index = df_path_top["Pathway name"]
    outcomes = list_form_to_onehot_form(df_path_top)
    symbols = pd.Series(outcomes.index, name="symbol")
    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)

    return


if __name__ == "__main__":
    main()
