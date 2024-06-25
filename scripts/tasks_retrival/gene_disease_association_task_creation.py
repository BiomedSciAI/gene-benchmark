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
import mygene
import pandas as pd
from task_retrieval import verify_source_of_data

from gene_benchmark.tasks import dump_task_definitions

DATA_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/parquet/"
)
TASK_NAME = "Gene Disease Association"
COLUMN_OF_SYMBOLS = "targetId"
COLUMN_OF_OUTCOME = "score"


def get_id_to_symbol_df(list_of_gene_metadata):
    """
        crate a data frame mapping from the gene metadata dataframe.
        remove duplicates if they exist, set the index to the "query",
        which holds the gene ids and return the mapping dataframe.
        this is a set of unique mappings.  It is used to map the ids in the data,
        which will contain duplicates, to the corresponding symbol names
    Args:
        list_of_gene_metadata (list): list containing gene metadata.

    Returns
    -------
        pd.DataFrame: a data frame with the gene id as index with the symbol as value

    """
    gene_metadata_df = pd.DataFrame(list_of_gene_metadata)
    # some target ids have multiple symbols
    gene_metadata_df = gene_metadata_df.drop_duplicates(subset="query")
    gene_metadata_df.index = gene_metadata_df["query"]
    return gene_metadata_df


def get_symbols(gene_targetId_list):
    """
        given s list of gene id's (names Like ENSG00000006468) this method
        uses the MyGenInfo package to retrieve the gene symbol (name like PLAC4).
        The retreaved dataframe is made into a translation df from id to metadata (symbol in this case),
        and is used to translate the original list ot a symbols of the same length, with the matching symbols
        in the corresponding locations.

    Args:
    ----
        gene_targetId_list (list): list of gene id's (names Like ENSG00000006468)

    Returns:
    -------
         pd.DataFrame: dataframe with symbols corresponding to the input

    """
    mg = mygene.MyGeneInfo()
    list_of_gene_metadata = mg.querymany(
        gene_targetId_list.drop_duplicates(), species="human", fields="symbol"
    )
    gene_metadata_df = get_id_to_symbol_df(list_of_gene_metadata)
    return pd.DataFrame(
        [gene_metadata_df.loc[x, "symbol"] for x in gene_targetId_list],
        columns=["symbol"],
    )


def get_gene_drug_association_data(input_file: str):
    """
    read data from open-targets' parquet data files, either directly or from a local file copy.
    The files are arranged with a fixed base name and part-index going from 0 to 199.  This is a standard way to save such data in parts
    and the total number of parts differes, so the code starts at 0 until it fails to read the file, and stops there.

    Args:
    ----
        input_file (str): base path (url or file path) of the parquet directory

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

    return pd.concat(res)


def extract_outcome_df(
    downloaded_dataframe: pd.DataFrame, target_column_name: str = COLUMN_OF_OUTCOME
) -> pd.DataFrame:
    outcomes = pd.Series(downloaded_dataframe[target_column_name], name="Outcomes")
    return outcomes.map(lambda x: x.strip() if isinstance(x, str) else x)


def report_numerical_task_task(
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
            downloaded_dataframe.groupby(entities_cols)["score"].mean().reset_index()
        )
    outcomes = extract_outcome_df(downloaded_dataframe)
    dump_task_definitions(
        entities_cols[entities_cols], outcomes, main_task_directory, task_name
    )

    if verbose:
        report_numerical_task_task(downloaded_dataframe, main_task_directory, task_name)


if __name__ == "__main__":
    main()
