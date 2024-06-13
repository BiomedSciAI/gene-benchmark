"""
Download disease parquet file from open targets
and save the description data
see further details at the following :
https://docs.mygene.info/en/latest/doc/query_service.html?highlight=gene%20Summary#query-parameters.

"""

import click
import pandas as pd


def download_gda_data():
    prq_location = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/23.09/output/etl/parquet/"
    res = []
    file_exist = True
    part_ind = 0
    while file_exist:
        try:
            link_ars = f"diseases/part-{part_ind:05}-e2a1713c-e8cb-4d40-96c5-ce4884c5a968-c000.snappy.parquet"
            gda_df = pd.read_parquet(prq_location + link_ars)
            res.append(gda_df)
            part_ind = part_ind + 1
        except:
            file_exist = False
            break
    if part_ind != 199:
        raise Warning(
            f"The script downloaded {part_ind} files check that the download all the files were downloaded "
        )

    return pd.concat(res)


@click.command()
@click.option(
    "--output-file-name",
    type=click.STRING,
    help="The output file name",
    default="disease_descriptions.csv",
)
@click.option(
    "--save-columns",
    type=click.STRING,
    help="The columns to save from the original data",
    default="id,name,ancestors,descendants,description",
)
def main(output_file_name, save_columns):
    disease_ass_df = download_gda_data()
    disease_ass_df[save_columns.split(",")].to_csv(output_file_name)


if __name__ == "__main__":
    main()
