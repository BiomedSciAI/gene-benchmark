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


def report_task_single_col(df, task_dir_name, task_name):
    print(f"Task {task_name} saved to {task_dir_name}/ \n")
    col_name = df.columns[0]
    print(df[col_name].value_counts().to_string())


def read_table(
    input_file: str | Path,
    allow_downloads: bool = False,
    strip_values: bool = True,
    **kwargs,
):
    try:
        #  "NA" is the symbol for "neuroacanthocytosis".  Unless the na_filter is turned off, it would be read as Nan
        downloaded_dataframe = pd.read_csv(input_file, **kwargs, na_filter=False)
        if strip_values:
            downloaded_dataframe = downloaded_dataframe.map(
                lambda x: x.strip() if type(x) == str else x
            )
    except Exception as exception:
        raise RuntimeError(f"could not read {input_file}") from exception
    return downloaded_dataframe
