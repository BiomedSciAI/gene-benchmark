import click
import mygene
import pandas as pd


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
    return [gene_metadata_df.loc[x, "symbol"] for x in gene_targetId_list]


def get_id_to_symbol_df(list_of_gene_metadata):
    """
        The method converts a list of gene metadata into a data frame,
        each dictionary will contain the field symbol and the gene id as the query value
    Args:
        list_of_gene_metadata (list): list containing gene metadata.

    Returns
    -------
        pd.DataFrame: a data frame with the gene id as index with the symbol as value

    """
    gene_metadata_df = pd.DataFrame(list_of_gene_metadata)
    # some target id have multiple symbols
    gene_metadata_df = gene_metadata_df.drop_duplicates(subset="query")
    gene_metadata_df.index = gene_metadata_df["query"]
    return gene_metadata_df


def download_gda_data():
    prq_location = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/parquet/"
    res = []
    file_exist = True
    part_ind = 0
    while file_exist:
        try:
            link_ars = f"associationByDatasourceDirect/part-{part_ind:05}-6866be1a-be5d-40cf-bdf6-627bef1d0410-c000.snappy.parquet"
            gda_df = pd.read_parquet(prq_location + link_ars)
            res.append(gda_df)
            part_ind = part_ind + 1
        except:
            file_exist = False
            break

    return pd.concat(res)


@click.command()
@click.option(
    "--output-file-name",
    type=click.STRING,
    help="The output file name",
    default="gene_disease_association.csv",
)
@click.option(
    "--association-type",
    type=click.STRING,
    help="The type of association to save",
    default="genetic_association",
)
def main(output_file_name, association_type):
    disease_ass_df = download_gda_data()
    gda_df = disease_ass_df.loc[disease_ass_df["datatypeId"] == association_type, :]
    print(f"Adding symbols for: {gda_df.shape[0]} associations")
    sym = get_symbols(gda_df["targetId"])
    gda_df.loc[:, "symbol"] = sym
    gda_df.to_csv(output_file_name)


if __name__ == "__main__":
    main()
