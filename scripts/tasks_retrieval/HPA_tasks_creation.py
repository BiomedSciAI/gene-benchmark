import click
import pandas as pd

from gene_benchmark.task_retrieval import (
    check_data_type,
    create_single_label_task,
    load_yaml_file,
    print_numerical_task_report,
    report_task_single_col,
    tag_list_to_multi_label,
)
from gene_benchmark.tasks import dump_task_definitions

COLUMN_TO_CLEAR_SEMICOLON = "Cell line expression cluster"
DATA_URL = "https://v23.proteinatlas.org/download/proteinatlas.tsv.zip"


def import_data(url):
    data = pd.read_csv(url, sep="\t")
    return data


def format_hpa_columns(data, clear_semicolon=None):
    pathology_columns = list(
        filter(lambda x: "Pathology prognostics" in x, data.columns)
    )
    data[pathology_columns] = data[pathology_columns].replace(
        r"\s*\(\d+\.?\d*e[+-]?\d*\)", "", regex=True
    )
    if not clear_semicolon is None and COLUMN_TO_CLEAR_SEMICOLON in data.columns:
        data[clear_semicolon] = (
            data[clear_semicolon].astype(str).apply(lambda x: x.replace(";", ""))
        )
    return data


def create_tasks(data, main_task_directory, verbose=False):
    for col in data:
        current_col_data = data[col]
        current_col_data = current_col_data.replace("", pd.NA)
        current_col_data = current_col_data.dropna()
        current_col_data = current_col_data.drop(
            current_col_data[current_col_data.index.str.contains("ENSG")].index
        )
        data_type = check_data_type(current_col_data)
        task_name = col.replace("/", "|")
        if data_type == "multi_class":
            entities, outcomes = tag_list_to_multi_label(current_col_data)
            if verbose:
                print(
                    f"Create task {task_name} at {main_task_directory} outcomes shaped {outcomes.shape}"
                )
        else:
            entities, outcomes = create_single_label_task(current_col_data)
            if data_type == "numerical" and verbose:
                print_numerical_task_report(outcomes, main_task_directory, task_name)
            elif verbose:
                report_task_single_col(outcomes, main_task_directory, task_name)
        dump_task_definitions(entities, outcomes, main_task_directory, task_name)


@click.command()
@click.option(
    "--columns-to-use-yaml",
    type=click.STRING,
    help="A path to a yaml file containing the column names to be used as tasks",
    default="scripts/tasks_retrieval/hpa_column_names_for_tasks.yaml",
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=True,
)
def main(
    columns_to_use_yaml, main_task_directory, allow_downloads, input_file, verbose
):
    if allow_downloads:
        data = import_data(DATA_URL)
    else:
        data = import_data(input_file)

    columns_to_use = load_yaml_file(columns_to_use_yaml)
    data = data.set_index("Gene")
    data = data[columns_to_use]

    data = format_hpa_columns(data, clear_semicolon=COLUMN_TO_CLEAR_SEMICOLON)

    create_tasks(data, main_task_directory, verbose)


if __name__ == "__main__":
    main()
