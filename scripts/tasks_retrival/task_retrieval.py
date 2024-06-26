import pickle
from io import BytesIO
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import mygene
import numpy as np
import pandas as pd
import requests
from yaml import safe_load


def verify_source_of_data(
    input_file: str | None, url: str | None, allow_downloads: bool = False
) -> str:
    """
    verify or provide source for data.  Data may be a local file, or if --allow-downloads is on, it will be
    the DATA_URL.
    This method will exit with an error if the input is not consistent with the workflow.

    Args:
    ----
        input_file (str | None): name if input file.  None if not set, non-url path if set.
        allow_downloads (bool, optional): has the user opted in to download the data from the source.
            Defaults to False.

    Returns:
    -------
        str: path to data file, wither local of the default.

    """
    if input_file is None:
        if not allow_downloads:
            raise ValueError(
                f"Please enter path to local file via --input-file or turn on --allow-downloads to download task source from {input_file}"
            )
        # input file not given, allow download on.
        return url
    elif allow_downloads:
        raise ValueError(
            "Arguments ambiguous:  Either give a local path of download from the web."
        )
    parsed_path = urlparse(str(input_file))
    if not parsed_path.netloc == "":
        raise ValueError(
            f'Input path "{input_file}" is not a local file.  Please enter pre-downloaded file path or allow download'
        )
    return input_file


def report_task_single_col(
    outcome_series: pd.Series, task_dir_name: str | Path, task_name: str
):
    """
    Reporting the class distribution for a single class prediction task.

    Args:
    ----
        outcome_series (pd.Series): The outcome
        task_dir_name (str | Path): the path in which the task is saved
        task_name (str): the name of the task

    """
    print(f"Task {task_name} saved to {task_dir_name}/ \n")
    print(outcome_series.value_counts().to_string())


def read_table(
    input_file: str | Path,
    strip_values: bool = True,
    filter_na: bool = False,
    **kwargs,
):
    """
    Reads a table from an input file.

    Args:
    ----
        input_file (str | Path): The location of the input file
        strip_values (bool, optional): Strip the strings of the table. Defaults to True.
        filter_na (bool, optional): "NA" is the symbol for "neuroacanthocytosis", Unless
        the na_filter is turned off, it would be read as Nan. Defaults to False.
        kwargs: To be transferred to the pandas read CSV method

    Raises:
    ------
        RuntimeError: If the table is unreadable

    Returns:
    -------
        pd.DataFrame: A data frame containing the table

    """
    try:
        downloaded_dataframe = pd.read_csv(input_file, **kwargs, na_filter=filter_na)
        if strip_values:
            downloaded_dataframe = downloaded_dataframe.map(
                lambda x: x.strip() if type(x) == str else x
            )
    except Exception as exception:
        raise RuntimeError(f"could not read {input_file}") from exception
    return downloaded_dataframe


def load_pickle_from_url(url):
    """
    Load a pickle file from a URL.

    Parameters
    ----------
    url (str): The URL of the pickle file.

    Returns
    -------
    object: The object loaded from the pickle file.

    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Create a BytesIO object from the response content
        file_object = BytesIO(response.content)

        # Load the pickle file from the BytesIO object
        data = pickle.load(file_object)

        return data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except pickle.UnpicklingError as e:
        print(f"Error loading the pickle file: {e}")


def get_symbols(gene_targetId_list):
    """
        given s list of gene id's (names Like ENSG00000006468) this method
        uses the MyGenInfo package to retrieve the gene symbol (name like PLAC4).

    Args:
    ----
        gene_targetId_list (list): list of gene id's (names Like ENSG00000006468)

    Returns:
    -------
        list: List of corresponding symbols

    """
    mg = mygene.MyGeneInfo()
    list_of_gene_metadata = mg.querymany(
        gene_targetId_list, species="human", fields="symbol"
    )
    gene_metadata_df = get_id_to_symbol_df(list_of_gene_metadata)
    symblist = [gene_metadata_df.loc[x, "symbol"] for x in gene_targetId_list]
    return [v for v in symblist if not pd.isna(v)]


def get_id_to_symbol_df(list_of_gene_metadata):
    """
        The method converts a list of gene metadata into a data frame,
        each dictionary will contain the field symbol and the gene id as the query value.

    Args:
    ----
        list_of_gene_metadata (list): list containing gene metadata.

    Returns:
    -------
        pd.DataFrame: a data frame with the gene id as index with the symbol as value

    """
    gene_metadata_df = pd.DataFrame(list_of_gene_metadata)
    # some target id have multiple symbols
    gene_metadata_df = gene_metadata_df.drop_duplicates(subset="query")
    gene_metadata_df.index = gene_metadata_df["query"]
    return gene_metadata_df


def print_numerical_task_report(
    outcomes: pd.Series, main_task_directory: str, task_name: str
):
    """
    prints a short task report.

    Args:
    ----
        df (pd.Series): _description_
        main_task_directory (str): the main folder for the task
        task_name (str): The task name

    """
    print(f"Task saved at {main_task_directory}  under {task_name} /\n")
    print(f"n = {len(outcomes)} mean {outcomes.mean():.2f} sd {outcomes.std():.2f}")


def list_form_to_onehot_form(
    list_df: pd.DataFrame,
    participant_col_name: str = "Submitted entities found",
    delimiter: str = ";",
) -> pd.DataFrame:
    """
    Give a pathway data frame that has each pathway as a row with
       a list of included genes the method creates a data frame where each
       row is a gene and each column is a pathway the cells are true when
       the gene is participating in the pathways.

    Args:
    ----
        pathway_df (pd.DataFrame): A data frame with pathways as rows and a gene in one of the cells
        pathway_name (str): The name of the pathways name columns
        included_genes (str): The name of the included genes in a pathway
        Submitted entities found with the participating genes

    Returns:
    -------
        pd.DataFrame: A one hot dataframe where rows are genes and columns are pathways

    """
    full_identifier_list = delimiter.join(list_df[participant_col_name].values).split(
        delimiter
    )
    unique_identifier_list = {x.strip() for x in full_identifier_list}
    onehot_df = pd.DataFrame(
        index=list(unique_identifier_list), columns=list_df.index, data=False
    )
    for pathway_idx in list_df.index:
        path_genes = list_df.loc[pathway_idx, participant_col_name].split(delimiter)
        onehot_df.loc[path_genes, pathway_idx] = True
    return onehot_df


def check_data_type(
    data_col: pd.Series,
    binary_name: str = "binary",
    category_name: str = "categorical",
    numerical_name: str = "numerical",
    multi_class_name: str = "multi_class",
    multi_regex: str = "[,;]",
) -> str:
    """
    Determines the column data type.

    Args:
    ----
        data_col (pd.Series): The series that is evaluated
        binary_name (str, optional): The string name to be used if the data is binary. Defaults to "binary".
        category_name (str, optional): The string name to be used if the data is categorical (multi-class). Defaults to 'categorical'.
        numerical_name (str, optional): The string name to be used if the data is numerical. Defaults to "numerical".
        multi_class_name (str, optional): The string name to be used if the data is multi class. Defaults to "multi_class".
        multi_regex (str, optional): If the values of the series contains the regex then it's a multi label . Defaults to "[,;]".

    Returns:
    -------
        str: the string type of the column

    """
    if data_col.nunique() == 2:
        return binary_name
    elif (data_col.nunique() > 2) & (data_col.dtypes == object):
        if data_col.astype(str).str.contains(multi_regex).any():
            return multi_class_name
        return category_name
    elif (data_col.nunique() > 2) & (
        (data_col.dtypes == "int64") | (data_col.dtypes == "float64")
    ):
        return numerical_name


def load_yaml_file(yaml_path: str):
    """
    loads a yaml file into an object.

    Args:
    ----
        yaml_path (str): the path to a yaml file

    Returns:
    -------
        object: the loaded yaml file

    """
    with open(yaml_path) as f:
        loaded_yaml = safe_load(f)
    return loaded_yaml


def create_single_label_task(
    current_col_data: pd.Series,
    entities_name: str = "symbol",
    outcomes_name: str = "Outcomes",
) -> tuple[pd.Series, pd.Series]:
    """
    take a series and creates task from the index and values.

    Args:
    ----
        current_col_data (pd.Series): the series to turn into a task
        entities_name (str, optional): the name to be used for the entities. Defaults to "symbol".
        outcomes_name (str, optional): the name to be used for the outcomes. Defaults to "Outcomes".

    Returns:
    -------
        tuple[pd.Series,pd.Series]: entities and outcomes series

    """
    entities = pd.Series(current_col_data.index, name=entities_name)
    outcomes = pd.Series(current_col_data.values, name=outcomes_name)
    return entities, outcomes


def tag_list_to_multi_label(
    current_col_data: pd.Series, entities_name: str = "symbol", delimiter: str = ","
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Takes a table with entities in rows and a tag cloud of attributes in values
      and converts into a multi label task.

    Args:
    ----
        current_col_data (pd.Series): Series with entities as indexes and the attribute list as the values
        entities_name (str, optional): Type of the entities. Defaults to 'symbol'.
        delimiter (str, optional): The delimiter in the attributes cloud . Defaults to ','.

    Returns:
    -------
        tuple[pd.Series,pd.DataFrame]: A tuple with the entities and a dta frame where each column
        is a attribute and the values represent the assignment to each attribute

    """
    split_values_df = current_col_data.apply(
        lambda x: [item.strip() for item in x.split(delimiter)]
    )
    vocab = list(set(np.concatenate(split_values_df.values)))
    outcome_df = pd.DataFrame(0, index=split_values_df.index, columns=vocab)
    for index in split_values_df.index:
        true_cat = split_values_df[index]
        if not isinstance(true_cat, list):
            true_cat = list(set(chain(*true_cat.values)))
        else:
            true_cat = list(set(true_cat))
        outcome_df.loc[index, true_cat] = 1
    entities = pd.Series(outcome_df.index, name=entities_name)
    return entities, outcome_df
