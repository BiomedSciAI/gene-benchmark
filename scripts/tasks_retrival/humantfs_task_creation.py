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

from pathlib import Path
from urllib.parse import urlparse

import click
import pandas as pd

DATA_URL = (
    "https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv"
)
TASK_NAME = "TF vs non-TF"
COLUMN_OF_SYMBOLS = "HGNC symbol"


def verify_source_of_data(input_file: str | None, allow_downloads: bool = False) -> str:
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
        return DATA_URL
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


def read_table(input_file: str | Path, allow_downloads: bool = False, **kwargs):
    try:
        #  "NA" is the symbol for "neuroacanthocytosis".  Unless the na_filter is turned off, it would be read as Nan
        downloaded_dataframe = pd.read_csv(input_file, **kwargs, na_filter=False)
    except Exception as exception:
        raise RuntimeError(f"could not read {input_file}") from exception
    return downloaded_dataframe


def extract_symbols_df(
    downloaded_dataframe: pd.DataFrame, column_of_symbols: str = COLUMN_OF_SYMBOLS
) -> pd.DataFrame:
    """Splice the column of the gene symbols into a new data frame and rename."""
    symbols = pd.DataFrame({"symbol": downloaded_dataframe[column_of_symbols]})
    return symbols.map(lambda x: x.strip() if type(x) == str else x)


def extract_outcome_df(
    downloaded_dataframe: pd.DataFrame, target_column_name: str = "Is TF?"
) -> pd.DataFrame:
    """Splice the two columns of interest into a new data frame, rename and extract HLC Class from the name."""
    outcomes = pd.DataFrame({"Outcomes": downloaded_dataframe[target_column_name]})
    # make sure there are no extra spaces around the values and return
    return outcomes.map(lambda x: x.strip() if type(x) == str else x)


def report_task(df, task_dir_name):
    print(f"Task saved to {task_dir_name}/")

    col_name = df.columns[0]
    print()  # for readability of the message
    print(df[col_name].value_counts().to_string())


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

    downloaded_dataframe = read_table(input_file=input_path_or_url)
    symbols = extract_symbols_df(downloaded_dataframe)
    outcomes = extract_outcome_df(downloaded_dataframe)

    # entered by the user.
    main_task_directory = Path(main_task_directory)

    # the specific directory for the task
    task_dir_name = main_task_directory / task_name
    task_dir_name.mkdir(exist_ok=True)

    # save symbols to CSV file
    symbols.to_csv(task_dir_name / "entities.csv", index=False)

    # save outcomes to CSV file
    outcomes.to_csv(task_dir_name / "outcomes.csv", index=False)

    if verbose:
        report_task(outcomes, task_dir_name)


if __name__ == "__main__":
    main()
