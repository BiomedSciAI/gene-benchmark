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

from pathlib import Path
from urllib.parse import urlparse

import click
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

DATA_URL = (
    "https://raw.githubusercontent.com/jingcheng-du/Gene2vec/master/predictionData"
)
TASK_NAME = "Gene2Gene"


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
    disable_na_values = {"NA"}
    my_default_na_values = STR_NA_VALUES - disable_na_values

    try:
        #  "NA" is the symbol for "neuroacanthocytosis".  Unless the na_filter is turned off, it would be read as Nan
        downloaded_dataframe = pd.read_csv(
            input_file, **kwargs, keep_default_na=False, na_values=my_default_na_values
        )
    except Exception as exception:
        raise RuntimeError(f"could not read {input_file}") from exception
    return downloaded_dataframe


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

    tmp_col_names = ["s1", "s2"]
    symbols = pd.concat(
        [
            read_table(
                f"{input_path_or_url}/{entity_file}",
                sep=" ",
                header=0,
                names=tmp_col_names,
            )
            for entity_file in entity_files
        ]
    )

    outcomes = pd.concat(
        [
            read_table(
                f"{input_path_or_url}/{outcome_file}",
                sep=" ",
                header=0,
                names=["outcomes"],
            )
            for outcome_file in outcome_files
        ]
    )

    assert len(symbols) == len(
        outcomes
    ), " the lengths of the concatinated files does not match"

    # clean extra space
    symbols = symbols.map(lambda x: x.strip())
    outcomes = outcomes.map(lambda x: x.strip() if type(x) == str else x)

    # removed all entities that are duplicated, which indicates an issue with the data curation
    if not keep_duplicates:
        duplicate_pairs = symbols.duplicated(keep=False)

        outcomes = outcomes.loc[~duplicate_pairs]
        symbols = symbols.loc[~duplicate_pairs]

    # rename the columns - seems safer not the try and manipulate data frames with non unique coloumn names
    symbols.columns = ["symbol", "symbol"]

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
