"""
Script for extracting the Gene2Vec prediction data.

Original data is downloaded from the Gene2vec project page at
https://github.com/jingcheng-du/Gene2vec/blob/master/README.md
via the supplied textual files that can be found at
https://github.com/jingcheng-du/Gene2vec/tree/master/predictionData

The data was released under the MIT License.
See https://github.com/jingcheng-du/Gene2vec/blob/master/LICENSE for full details


In citation of this data, please site the original authors:

Du, J., Jia, P., Dai, Y. et al.
Gene2vec: distributed representation of genes based on co-expression.
BMC Genomics 20 (Suppl 1), 82 (2019).
https://doi.org/10.1186/s12864-018-5370-x

"""

import click
import pandas as pd
from task_retrieval import read_table, report_task_single_col, verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions

DATA_URL = (
    "https://raw.githubusercontent.com/jingcheng-du/Gene2vec/master/predictionData"
)
TASK_NAME = "Gene2Gene"


def concat_tables(input_path_or_url, entity_files, tmp_col_names, sep=" ", header=0):
    return pd.concat(
        [
            read_table(
                f"{input_path_or_url}/{entity_file}",
                sep=sep,
                header=header,
                names=tmp_col_names,
            )
            for entity_file in entity_files
        ]
    )


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--task-name",
    "-n",
    type=click.STRING,
    help="name for the specific task",
    default=TASK_NAME,
)
@click.option(
    "--input-file",
    "-i",
    type=click.STRING,
    help="path local data directory containing the files.",
    default=None,
)
@click.option(
    "--allow-downloads",
    help=f"download files directly from {DATA_URL}",
    type=click.BOOL,
    default=False,
)
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--keep-duplicates",
    is_flag=True,
    default=True,
    help="Do not remove entities that appear more than once",
)
@click.option(
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
def main(
    task_name,
    main_task_directory,
    input_file,
    allow_downloads,
    keep_duplicates,
    verbose,
):
    entity_files = [
        "train_text.txt",
        "valid_text.txt",
        "test_text.txt",
    ]
    outcome_files = [
        "train_label.txt",
        "valid_label.txt",
        "test_label.txt",
    ]

    input_path_or_url = verify_source_of_data(
        input_file, allow_downloads=allow_downloads
    )
    symbols = concat_tables(input_path_or_url, entity_files, ["s1", "s2"])
    outcomes = concat_tables(input_path_or_url, outcome_files, ["outcomes"])

    assert len(symbols) == len(
        outcomes
    ), " the lengths of the concatinated files does not match"

    # removed all entities that are duplicated, which indicates an issue with the data curation
    if not keep_duplicates:
        duplicate_pairs = symbols.duplicated(keep=False)

        outcomes = outcomes.loc[~duplicate_pairs]
        symbols = symbols.loc[~duplicate_pairs]

    # rename the columns - seems safer not the try and manipulate data frames with non unique coloumn names
    symbols.columns = ["symbol", "symbol"]

    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)
    if verbose:
        report_task_single_col(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
