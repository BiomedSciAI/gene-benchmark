from os import makedirs
from pathlib import Path

import click
import pandas as pd
import requests


def get_symbol_list(
    url="https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/json/hgnc_complete_set.json",
):
    with requests.get(url) as response:
        response.raise_for_status()
        reactome_res = response.json()
    return [v["symbol"] for v in reactome_res["response"]["docs"]]


def get_token(
    token_list,
    projection_url="https://reactome.org/AnalysisService/identifiers/projection",
):
    headers = {
        "Accept": "application/json",
        "Content-Type": "text/plain",
    }
    symbols = "\n".join(token_list)
    response = requests.post(
        projection_url,
        headers=headers,
        data=symbols,
    )
    return response.json()["summary"]["token"]


def get_top_level_pathway(
    url="https://reactome.org/download/current/ReactomePathwaysRelation.txt",
):
    hierarchies_df = pd.read_csv(
        url, delimiter="\t", header=0, names=["parent", "child"]
    )
    pathway_that_are_parents = set(hierarchies_df["parent"].values)
    pathway_that_are_children = set(hierarchies_df["child"].values)
    pathway_who_are_just_parents = pathway_that_are_parents - pathway_that_are_children
    return pathway_who_are_just_parents


def pathway_to_onehot(pathway_df):
    any_pathway_genes = list(
        set(";".join(pathway_df["Submitted entities found"].values).split(";"))
    )
    outcomes = pd.DataFrame(
        index=any_pathway_genes, columns=pathway_df["Pathway name"], data=False
    )
    for pathway_idx in pathway_df.index:
        path_genes = pathway_df.loc[pathway_idx, "Submitted entities found"].split(";")
        pathway_name = pathway_df.loc[pathway_idx, "Pathway name"]
        outcomes.loc[path_genes, pathway_name] = True
    return outcomes


def dump_to_task(task_dir, outcomes_df):
    entities_path = task_dir / "entities.csv"
    outcomes_path = task_dir / "outcomes.csv"
    pd.Series(outcomes_df.index, name="symbol").to_csv(entities_path, index=False)
    outcomes_df.to_csv(outcomes_path, index=False)


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
    default="",
)
@click.option(
    "--top-pathways-file",
    type=click.STRING,
    help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
    default="",
)
def main(
    main_task_directory, task_name, allow_downloads, pathways_file, top_pathways_file
):
    if allow_downloads:
        symb_list = get_symbol_list()
        token = get_token(symb_list)
        url = f"https://reactome.org/AnalysisService/download/{token}/pathways/TOTAL/result.csv"
        df_path = pd.read_csv(url, index_col="Pathway identifier")
        top_level = get_top_level_pathway()
    else:
        df_path = pd.read_csv(pathways_file)
        top_level = pd.read_csv(top_pathways_file)

    top_in_file_paths = top_level.intersection(set(df_path.index))
    df_path_top = df_path.loc[list(top_in_file_paths), :]
    outcomes = pathway_to_onehot(df_path_top)
    task_dir = Path(main_task_directory) / f"{task_name}"
    makedirs(task_dir, exist_ok=True)
    dump_to_task(task_dir, outcomes)
    return


if __name__ == "__main__":
    main()
