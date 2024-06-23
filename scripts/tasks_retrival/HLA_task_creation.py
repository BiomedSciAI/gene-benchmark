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

Script for extracting the gene symbol to HLA class I/class II task.

Original data is downloaded from the HGNC project page at
https://www.genenames.org/data/genegroup/#!/group/588
via the supplied textual file that can be found at
https://www.genenames.org/cgi-bin/genegroup/download?id=588&type=node

The data was released under the Creative Commons Public Domain (CC0) License.
original license statment can be found in https://www.genenames.org/about/license/


In citation of this data, please site the original authors:

Seal RL, Braschi B, Gray K, Jones TEM, Tweedie S, Haim-Vilmovsky L, Bruford EA.
Genenames.org: the HGNC resources in 2023.
Nucleic Acids Res. 2023 Jan 6;51(D1):D1003-D1009.
doi: 10.1093/nar/gkac888. PMID: 36243972; PMCID: PMC9825485.

"""

import click
import pandas as pd
from task_retrival import read_table, report_task_single_col, verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions

DATA_URL = "https://www.genenames.org/cgi-bin/genegroup/download?id=588&type=node"

ENTITIES_COL = "Approved symbol"


def extract_symbols_df(
    downloaded_dataframe: pd.DataFrame, column_of_symbols: str = ENTITIES_COL
) -> pd.DataFrame:
    """Splice the column of the gene symbols into a new data frame and rename."""
    symbols = pd.DataFrame({"symbol": downloaded_dataframe[column_of_symbols]})
    return symbols.map(lambda x: x.strip() if type(x) == str else x)


def extract_HLA_class_df(
    downloaded_dataframe: pd.DataFrame, target_column_name: str = "Approved name"
) -> pd.DataFrame:
    """Splice the two columns of interest into a new data frame, rename and extract HLC Class from the name."""
    # extract and rename to the column to the standard name
    # class is extracted from the end of the approved name field, after the comma.
    # class is either "class I" or "class II"
    outcomes = pd.DataFrame(
        {
            "Outcomes": downloaded_dataframe[target_column_name].map(
                lambda x: x.split(",")[1]
            ),
        }
    )
    # make sure there are no extra spaces around the values and return
    return outcomes.map(lambda x: x.strip() if type(x) == str else x)


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--task-name",
    "-n",
    type=click.STRING,
    help="name for the specific task",
    default="HLA class I vs class II",
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
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
def main(task_name, main_task_directory, input_file, allow_downloads, verbose):
    input_path_or_url = verify_source_of_data(
        input_file, allow_downloads=allow_downloads
    )

    # downloaded_dataframe = read_table(input_file=input_path_or_url)
    downloaded_df = read_table(input_file=input_path_or_url, sep="\t")
    symbols = downloaded_df[ENTITIES_COL]
    outcomes = extract_HLA_class_df(downloaded_df)
    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)
    if verbose:
        report_task_single_col(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
