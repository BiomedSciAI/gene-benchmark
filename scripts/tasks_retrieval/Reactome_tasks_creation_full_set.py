import click
import pandas as pd
from task_retrieval import verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrival.task_retrieval import list_form_to_onehot_form

TOP_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def get_pathways_data(url):
    pathways_file = pd.read_csv(url, on_bad_lines="skip", header=None, delimiter="\t")
    pathways_data = pd.DataFrame(columns=["Pathway name", "idx", "symbol"])
    pathways_data[["Pathway name", "idx"]] = pathways_file.iloc[:, :2]
    pathways_data["symbol"] = pathways_file.iloc[:, 2:].apply(
        lambda x: ",".join([str(v) for v in set(x) if not pd.isna(v)]), axis=1
    )
    pathways_data = pathways_data.set_index("idx")
    return pathways_data


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
    "--allow-downloads",
    type=click.BOOL,
    help="If false data files will be downloaded directly from HGNC and reactome, set to true only if you trust the urls above",
    default=False,
)
@click.option(
    "--pathways-file",
    type=click.STRING,
    help=f"A file with the pathway and the included symbol in {PATHWAYS_URL} format",
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
@click.option(
    "--add-top-pathways",
    type=click.BOOL,
    default=True,
)
def main(
    main_task_directory,
    allow_downloads,
    pathways_file,
    pathways_relation_file,
    verbose,
    add_top_pathways,
):

    pathways_relation_file = verify_source_of_data(
        pathways_relation_file, url=TOP_PATHWAYS_URL, allow_downloads=allow_downloads
    )
    hierarchies_df = pd.read_csv(
        pathways_relation_file, delimiter="\t", header=0, names=["parent", "child"]
    )
    pathways_file = verify_source_of_data(
        pathways_file, url=PATHWAYS_URL, allow_downloads=allow_downloads
    )
    pathways_df = get_pathways_data(pathways_file)
    if add_top_pathways:
        symbols, outcomes = create_top_level_task(hierarchies_df, pathways_df)
        dump_task_definitions(symbols, outcomes, main_task_directory, "top pathways")
    if verbose:
        print(f"{5} was created at {main_task_directory} shaped {outcomes.shape}")
    return


if __name__ == "__main__":
    main()
