"""
Script for extracting the Gene Disease assosiaction task.



Original data is downloaded from the Open Tragets page at
https://platform-docs.opentargets.org/
via the supplied parquet files that can be found at
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/parquet/

Data was provided by the autors under CC0 1.0 licence,
"This dedicates the data to the public domain, allowing downstream users to consume the data without restriction."
see https://platform-docs.opentargets.org/licence for full licence details.

In citation of this data, please site the original authors latest publication:

 David Ochoa, Andrew Hercules, Miguel Carmona, Daniel Suveges, Jarrod Baker, Cinzia Malangone,
 Irene Lopez, Alfredo Miranda, Carlos Cruz-Castillo, Luca Fumis, Manuel Bernal-Llinares, Kirill Tsukanov,
 Helena Cornu, Konstantinos Tsirigos, Olesya Razuvayevskaya, Annalisa Buniello, Jeremy Schwartzentruber,
 Mohd Karim, Bruno Ariano, Ricardo Esteban Martinez Osorio, Javier Ferrer, Xiangyu Ge,
 Sandra Machlitt-Northen, Asier Gonzalez-Uriarte, Shyamasree Saha, Santosh Tirunagari, Chintan Mehta,
 Juan María Roldán-Romero, Stuart Horswell, Sarah Young, Maya Ghoussaini, David G Hulcoop, Ian Dunham,
 Ellen M McDonagh,

 The next-generation Open Targets Platform: reimagined, redesigned, rebuilt,
 Nucleic Acids Research, Volume 51, Issue D1,
 6 January 2023, Pages D1353–D1359,
 https://doi.org/10.1093/nar/gkac1046


Taken from:
    https://platform-docs.opentargets.org/citation#latest-publication

"""

import click
import pandas as pd
from task_retrieval import get_symbols, verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions
from scripts.tasks_retrieval.task_retrieval import print_numerical_task_report

DATA_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/parquet/"
)
TASK_NAME = "Gene Disease Association"
COLUMN_OF_SYMBOLS = "targetId"
COLUMN_OF_OUTCOME = "score"


def get_gene_drug_association_data(
    input_file: str, strip_df: bool = True
) -> pd.DateFrame:
    """
    read data from open-targets' parquet data files, either directly or from a local file copy.
    The files are arranged with a fixed base name and part-index going from 0 to 199.  This is a standard way to save such data in parts
    and the total number of parts differs, so the code starts at 0 until it fails to read the file, and stops there.

    Args:
    ----
        input_file (str): base path (url or file path) of the parquet directory
        input_file (bool): if so strip the dataframe or not. Default: True

    Raises:
    ------
        RuntimeError: if no files had could be loaded

    Returns:
    -------
        pd.Dataframe: concatenated dataframe containing all the loaded data from all files

    """
    res = []
    file_exist = True
    part_ind = 0
    while file_exist:
        try:
            link_ars = f"associationByOverallDirect/part-{part_ind:05}-f96bc7c3-79fa-4d2a-8c16-5dfe4f3b853d-c000.snappy.parquet	"
            gda_df = pd.read_parquet(input_file + link_ars)
            res.append(gda_df)
            part_ind = part_ind + 1
        except Exception as exception:
            if not len(res):
                raise RuntimeError(f"could not read from {input_file}") from exception
            file_exist = False
            break

    total_df = pd.concat(res)
    if strip_df:
        total_df = total_df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return total_df


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
    "--average-duplicates",
    help="Due to conversion from ensemble id to symbol multiple entries might occur with the same symbol, this will average their association score",
    type=click.BOOL,
    default=True,
)
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
)
def main(
    task_name,
    input_file,
    allow_downloads,
    main_task_directory,
    average_duplicates,
    verbose,
):
    input_path_or_url = verify_source_of_data(
        input_file, url=DATA_URL, allow_downloads=allow_downloads
    )

    if verbose:
        print(f"creating the {task_name} task")
        print("This may take several minutes")
    downloaded_dataframe = get_gene_drug_association_data(input_file=input_path_or_url)

    downloaded_dataframe["symbol"] = get_symbols(
        downloaded_dataframe[COLUMN_OF_SYMBOLS]
    )
    entities_cols = ["symbol", "diseaseId"]
    if average_duplicates:
        downloaded_dataframe = (
            downloaded_dataframe.groupby(entities_cols)[COLUMN_OF_OUTCOME]
            .mean()
            .reset_index()
        )
    outcomes = pd.Series(downloaded_dataframe[COLUMN_OF_OUTCOME], name="Outcomes")
    dump_task_definitions(
        entities_cols[entities_cols], outcomes, main_task_directory, task_name
    )

    if verbose:
        print_numerical_task_report(outcomes, main_task_directory, task_name)


if __name__ == "__main__":
    main()
