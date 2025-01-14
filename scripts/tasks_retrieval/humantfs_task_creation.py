"""
(C) Copyright 2024 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Jun 11, 2024
Script for extracting the The Human Transcription Factors task.

Original data is downloaded from the The Human Transcription Factors page at
https://humantfs.ccbr.utoronto.ca/download.php
via the supplied CSV file that can be found at
https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv

In citation of this data, please site the original authors:

Lambert SA, Jolma A, Campitelli LF, Das PK, Yin Y, Albu M, Chen X, Taipale J, Hughes TR, Weirauch MT.(2018)
The Human Transcription Factors.
Cell. 172(4):650-665.
doi: 10.1016/j.cell.2018.01.029. Review.

"""

import click

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrieval.task_retrieval import (
    read_table,
    report_task_single_col,
    verify_source_of_data,
)

DATA_URL = "http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"
ENTITIES_COL = "HGNC symbol"


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--task-name",
    "-n",
    type=click.STRING,
    help="name for the specific task",
    default="TF vs non-TF",
)
@click.option(
    "--input-file",
    "-i",
    type=click.STRING,
    help="path local data file.",
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
    "--outcome-column",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="Is TF?",
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
    outcome_column,
    verbose,
):
    input_path_or_url = verify_source_of_data(
        input_file, url=DATA_URL, allow_downloads=allow_downloads
    )

    downloaded_df = read_table(input_file=input_path_or_url)
    symbols = downloaded_df[ENTITIES_COL]
    outcomes = downloaded_df[outcome_column]
    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)

    if verbose:
        report_task_single_col(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
