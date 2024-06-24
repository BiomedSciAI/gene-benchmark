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

Original data is downloaded from the Geneformer corpus
https://huggingface.co/datasets/ctheodoris/Genecorpus-30M

In citation of this data, please site the original authors:

Theodoris CV*, Xiao L, Chopra A, Chaffin MD, Al Sayed ZR, Hill MC,
Mantineo H, Brydon EM, Zeng Z, Liu XS, Ellinor PT*. Transfer learning
enables predictions in network biology. Nature. 2023 May 31;
Epub ahead of print. (*co-corresponding authors)

"""

from pathlib import Path

import click
import pandas as pd
from task_retrieval import (
    get_symbols,
    load_pickle_from_url,
    report_task_single_col,
    verify_source_of_data,
)

from gene_benchmark.tasks import dump_task_definitions

DATA_FILE_NAMES = {
    "bivalent_vs_lys4_only": "bivalent_promoters/bivalent_vs_lys4_only.pickle?download=true",
    "dosage sensitive vs insensitive TF": "dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle?download=true",
    "long vs short range TF": "tf_regulatory_range/tf_regulatory_range.pickle?download=true",
    "N1 targets": "notch1_network/n1_network.pickle?download=true",
    "N1 network": "notch1_network/n1_target.pickle?download=true",
    "bivalent vs non-methylated": "bivalent_promoters/bivalent_vs_no_methyl.pickle?download=true",
}
DATA_LOCAL_FILE_NAMES = {
    "bivalent_vs_lys4_only": Path("bivalent_promoters")
    / Path("bivalent_vs_lys4_only.pickle"),
    "dosage sensitive vs insensitive TF": Path("dosage_sensitive_tfs")
    / Path("dosage_sensitivity_TFs.pickle"),
    "long vs short range TF": Path("tf_regulatory_range")
    / Path("tf_regulatory_range.pickle"),
    "N1 targets": Path("notch1_network") / Path("n1_network.pickle"),
    "N1 network": Path("notch1_network") / Path("n1_target.pickle?"),
    "bivalent vs non-methylated": Path("bivalent_promoters")
    / Path("bivalent_vs_no_methyl.pickle"),
}
DATA_URL = "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main/example_input_files/gene_classification/"


def create_symbol_dict(data_dict):
    return {key: get_symbols(value) for (key, value) in data_dict.items()}


def dictionary_to_task(data_dict: dict, remove_duplicates: True):
    """
    Takes a dictionary containing gene labels as keys and a list
    of gene as a value and format them as a data frame where the symbols are
    the entities and the keys are the labels.

    Args:
    ----
        data_dict (dict): a dictionary with labels for keys and the corresponding entities as values
        remove_duplicates (bool, optional): remove duplicates symbols (keeps the first). Defaults to True.

    Returns:
    -------
        (pd.Series,pd.Series): A tuple containing the entities and outcomes as pd.Series.

    """
    ent_list = []
    out_list = []
    for key, value in data_dict.items():
        ent_list.extend(value)
        out_list.extend([key] * len(value))
    entities = pd.Series(ent_list, name="symbol")
    outcomes = pd.Series(out_list, name="Outcomes")
    if remove_duplicates:
        return entities[~entities.duplicated()], outcomes[~entities.duplicated()]
    else:
        return entities, outcomes


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--input-file",
    "-i",
    type=click.STRING,
    help="path local data file.",
    default=None,
)
@click.option(
    "--allow-downloads",
    type=click.BOOL,
    help=f"download files directly from {DATA_URL}",
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
    "--remove-duplicates",
    "-d",
    type=click.BOOL,
    help="Wether or not to remove duplicates from the task",
    default="True",
)
@click.option(
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
def main(
    input_file,
    allow_downloads,
    main_task_directory,
    remove_duplicates,
    verbose,
):
    for task_name, task_file in DATA_FILE_NAMES.items():
        input_path_or_url = verify_source_of_data(
            input_file, url=DATA_URL, allow_downloads=allow_downloads
        )
        if allow_downloads:
            full_path = input_path_or_url + task_file
        else:
            full_path = Path(input_path_or_url) / DATA_LOCAL_FILE_NAMES[task_name]

        data = load_pickle_from_url(full_path)
        symbols, outcomes = dictionary_to_task(
            create_symbol_dict(data), remove_duplicates=remove_duplicates
        )
        dump_task_definitions(symbols, outcomes, main_task_directory, task_name)
        if verbose:
            report_task_single_col(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
