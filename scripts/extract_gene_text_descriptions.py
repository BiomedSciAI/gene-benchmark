import logging
from itertools import product
from pathlib import Path

import click
import pandas as pd
import yaml

import gene_benchmark
import gene_benchmark.descriptor
import gene_benchmark.encoder
from gene_benchmark.descriptor import NaiveDescriptor
from gene_benchmark.deserialization import load_class

logger = logging.getLogger()
FORMAT = (
    "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
)
logging.basicConfig(format=FORMAT, level=logging.INFO)


def get_descriptor(model_dict: dict) -> gene_benchmark.descriptor.TextDescriptor:
    """
    Load descriptor from model dict.

    Args:
    ----
        model_dict (dict): model dict

    Returns:
    -------
        gene_benchmark.descriptor.TextDescriptor: a descriptor object

    """
    if "descriptor" in model_dict:
        descriptor = load_class(**model_dict["descriptor"])
    else:
        descriptor = NaiveDescriptor()
    return descriptor


def parse_gene_symbols(gene_symbols: str | Path) -> tuple[list[str], str]:
    """
    Parse gene symbols from request.

    Args:
    ----
        gene_symbols (Path): path to yaml of gene symbols


    Returns:
    -------
        tuple[list[str], str]: list of genes, name to use for gene list

    """
    with open(gene_symbols) as f:
        gene_symbol_list = yaml.safe_load(f)
    gene_symbols_name = gene_symbols.stem

    return gene_symbol_list, gene_symbols_name


def do_gene_descriptions(
    extraction_config: str | Path,
    gene_symbols: str | Path,
    output_folder: str | Path,
    allow_missing: bool,
    output_identifier: str,
) -> None:
    """
    Create descriptions for genes and generate embeddings from text model.

    Model is specified in extraction_config, genes in gene_symbols.

    Args:
    ----
        extraction_config (str | Path): path to extraction_config yaml with model details etc
        gene_symbols (str | Path): path to yaml with gene symbols
        output_folder (str | Path): folder to store results
        allow_missing (bool): whether to allow missing genes
        output_identifier (str): string to append to output file name (before suffix)

    """
    logger.info(f"Executing {extraction_config}, {gene_symbols}")
    extraction_config = Path(extraction_config)
    output_folder = Path(output_folder)
    gene_symbols = Path(gene_symbols)
    model_name = extraction_config.stem

    with open(extraction_config) as f:
        model_dict = yaml.safe_load(f)
    descriptor = get_descriptor(model_dict)
    gene_symbol_list, gene_symbols_name = parse_gene_symbols(gene_symbols)

    logger.info(f"Getting descriptions for {len(gene_symbol_list)} symbols")
    descriptions = descriptor.describe(
        pd.Series(gene_symbol_list, index=gene_symbol_list), allow_missing=allow_missing
    )
    descriptions_ofname = (
        output_folder
        / f"descriptions_{gene_symbols_name}_{model_name}{output_identifier}.csv"
    )
    descriptions.to_csv(descriptions_ofname)


@click.command()
@click.option(
    "--extraction-configs",
    "-e",
    type=click.STRING,
    help="config yaml with the descriptor and encoder for the extraction",
    multiple=True,
)
@click.option(
    "--output-folder",
    type=click.STRING,
    help="Output file to save the embeddings into",
    default="",
)
@click.option(
    "--gene-symbols",
    "-t",
    type=click.STRING,
    help="Path to .yaml yaml file that contains a list of gene symbol of type scope (default symbol)",
    required=True,
    multiple=True,
)
@click.option(
    "--allow-missing",
    type=click.BOOL,
    help="Allow missing prompts to be encoded",
    default=True,
)
@click.option(
    "--file-identifier",
    help="identifier inserted into output file name (before the extension)",
    type=click.STRING,
    default="",
)
def main(
    extraction_configs,
    output_folder,
    gene_symbols,
    allow_missing,
    file_identifier,
):
    for job in product(extraction_configs, gene_symbols):
        extraction_config, gene_symbol = job
        do_gene_descriptions(
            extraction_config,
            gene_symbol,
            output_folder,
            allow_missing,
            file_identifier,
        )


if __name__ == "__main__":
    main()
