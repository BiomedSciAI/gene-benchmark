from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


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
        filter_na (bool, optional): "NA" is the symbol for "neuroacanthocytosis", Unless the na_filter is turned off, it would be read as Nan. Defaults to False.
        kwargs: To be transferred to the pandas read CSV method

    Raises:
    ------
        RuntimeError: If the table is unreadable

    Returns:
    -------
        pd.DataFrame: A data frame containing the table

    """
    try:
        downloaded_dataframe = pd.read_csv(input_file, **kwargs, na_filter=False)
        if strip_values:
            downloaded_dataframe = downloaded_dataframe.map(
                lambda x: x.strip() if type(x) == str else x
            )
    except Exception as exception:
        raise RuntimeError(f"could not read {input_file}") from exception
    return downloaded_dataframe
