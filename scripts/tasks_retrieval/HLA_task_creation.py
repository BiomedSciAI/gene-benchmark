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
original license statement can be found in https://www.genenames.org/about/license/


In citation of this data, please site the original authors:

Seal RL, Braschi B, Gray K, Jones TEM, Tweedie S, Haim-Vilmovsky L, Bruford EA.
Genenames.org: the HGNC resources in 2023.
Nucleic Acids Res. 2023 Jan 6;51(D1):D1003-D1009.
doi: 10.1093/nar/gkac888. PMID: 36243972; PMCID: PMC9825485.

"""

import click

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrieval.task_retrieval import (
    read_table,
    report_task_single_col,
    verify_source_of_data,
)

DATA_URL = "https://www.genenames.org/cgi-bin/genegroup/download?id=588&type=node"

ENTITIES_COL = "Approved symbol"
OUTCOMES_COL = "Approved name"


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
        input_file, url=DATA_URL, allow_downloads=allow_downloads
    )
    downloaded_df = read_table(input_file=input_path_or_url, sep="\t")
    symbols = downloaded_df[ENTITIES_COL]
    outcomes = downloaded_df[OUTCOMES_COL].apply(lambda x: x.split(",")[1])
    dump_task_definitions(symbols, outcomes, main_task_directory, task_name)
    if verbose:
        report_task_single_col(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
