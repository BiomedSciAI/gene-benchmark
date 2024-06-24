import click
import numpy as np
import pandas as pd

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrival.task_retrieval import (
    check_data_type,
    create_single_label_task,
    load_yaml_file,
)

CELL_LINE = "Cell line expression cluster"
DATA_URL = "https://v23.proteinatlas.org/download/proteinatlas.tsv.zip"


def import_data(url):
    data = pd.read_csv(url, sep="\t")
    return data


def format_pathology_columns(data):
    pathology_columns = list(
        filter(lambda x: "Pathology prognostics" in x, data.columns)
    )
    data[pathology_columns] = data[pathology_columns].replace(
        r"\s*\(\d+\.?\d*e?-?\d*\)", "", regex=True
    )
    return data


def create_tasks(data, main_task_directory):
    for col in data:
        current_col_data = data[col]
        current_col_data = current_col_data.replace("", pd.NA)
        current_col_data = current_col_data.dropna()
        current_col_data = current_col_data.drop(
            current_col_data[current_col_data.index.str.contains("ENSG")].index
        )
        data_type = check_data_type(current_col_data)
        if data_type == "multi_class":
            entities, outcomes = create_multi_label_task(current_col_data)
        else:
            entities, outcomes = create_single_label_task(current_col_data)
        task_name = col.replace("/", "|")
        dump_task_definitions(entities, outcomes, main_task_directory, task_name)


def create_multi_label_task(current_col_data):
    split_values_df = current_col_data.apply(
        lambda x: [item.strip() for item in x.split(",")]
    )
    vocab = list(set(np.concatenate(split_values_df.values)))
    outcome_df = pd.DataFrame(0, index=split_values_df.index, columns=vocab)
    for index in range(split_values_df.shape[0]):
        outcome_df.iloc[index][split_values_df.iloc[index]] = 1
    entities = pd.Series(outcome_df.index, name="symbol")
    return entities, outcome_df


@click.command()
@click.option(
    "--columns-to-use-yaml",
    type=click.STRING,
    help="A path to a yaml file containing the column names to be used as tasks",
    default="scripts/tasks_retrival/hpa_column_names_for_tasks.yaml",
)
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--allow-downloads",
    "-l",
    type=click.BOOL,
    help=f"download files directly from {DATA_URL}",
    default=False,
)
@click.option(
    "--input-file", type=click.STRING, help="The path to the data file", default=None
)
def main(columns_to_use_yaml, main_task_directory, allow_downloads, input_file):
    if allow_downloads:
        data = import_data(DATA_URL)
    else:
        data = import_data(input_file)

    columns_to_use = load_yaml_file(columns_to_use_yaml)
    data = data.set_index("Gene")
    data = data[columns_to_use]

    data = format_pathology_columns(data)
    if CELL_LINE in data.columns:
        data[CELL_LINE] = (
            data[CELL_LINE].astype(str).apply(lambda x: x.replace(";", ""))
        )
    create_tasks(data, main_task_directory)


if __name__ == "__main__":
    main()
