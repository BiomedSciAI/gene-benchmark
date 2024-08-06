import warnings
from dataclasses import dataclass

import click
import pandas as pd

from gene_benchmark.task_retrieval import (
    list_form_to_onehot_form,
    verify_source_of_data,
)
from gene_benchmark.tasks import dump_task_definitions

HIERARCHIES_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
GENE_LIST_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def get_gene_list(url):
    gene_list_file = pd.read_csv(url, on_bad_lines="skip", header=None, delimiter="\t")
    pathway_to_gene_df = pd.DataFrame(
        columns=["pathway_description", "pathway", "genes"]
    )
    pathway_to_gene_df[["pathway_description", "pathway"]] = gene_list_file.iloc[:, :2]
    pathway_to_gene_df["genes"] = gene_list_file.iloc[:, 2:].apply(
        lambda x: [str(v) for v in set(x) if not pd.isna(v)], axis=1
    )
    pathway_to_gene_df = pathway_to_gene_df.set_index("pathway")
    return pathway_to_gene_df


def get_hierarchy_data(url):
    hierarchy_df = pd.read_csv(
        url, header=None, delimiter="\t", names=["parent", "child"]
    )
    hierarchy_df = hierarchy_df.set_index("parent")

    hierarchies = (
        hierarchy_df.groupby(hierarchy_df.index)["child"].apply(list).to_dict()
    )
    return hierarchies


def pathways_2_one_hot(pathways, path_2_gene):
    task_df = pd.DataFrame(columns=["genes"], index=pathways)
    task_df["genes"] = [
        ";".join(path_2_gene.populate_pathway(path)) for path in task_df.index
    ]
    return list_form_to_onehot_form(
        task_df, participant_col_name="genes", delimiter=";"
    )


@dataclass
class PathwaySeeks:
    pathway_to_gene: dict[str, list[str]]
    hierarchies: dict[str, list[str]]

    def get_genes(self, pathway: list[str]) -> list[str]:
        sub_genes = set()
        if pathway in self.pathway_to_gene:
            return self.pathway_to_gene[pathway]
        elif not pathway in self.hierarchies:
            warnings.warn(f"Pathway {pathway} has no sub pathways and no genes defined")
        else:
            for sub_pathways in self.hierarchies[pathway]:
                sub_genes.update(self.populate_pathway(sub_pathways))
        self.pathway_to_gene[pathway] = list(sub_genes)
        return list(sub_genes)


@click.command()
@click.option(
    "--main-task-directory",
    "-m",
    type=click.STRING,
    help="The task root directory.  Will not be created.",
    default="./tasks",
)
@click.option(
    "--allow-downloads",
    type=click.BOOL,
    help="If false data files will be downloaded directly from HGNC and reactome, set to true only if you trust the urls above",
    default=False,
)
@click.option(
    "--pathway-identifier",
    type=click.STRING,
    help="Pathway identifier from which we want to create multilabel task.",
    required=True,
    multiple=True,
)
@click.option(
    "--hierarchy-file",
    type=click.STRING,
    help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
    default=None,
)
@click.option(
    "--pathway-to-gene-file",
    "-m",
    type=click.STRING,
    help="A file with the pathways and the genes involved in the pathways.",
    default=None,
)
@click.option(
    "--verbose/--quite",
    "-v/-q",
    is_flag=True,
    default=True,
)
def main(
    main_task_directory,
    allow_downloads,
    pathway_identifier,
    hierarchy_file,
    pathway_to_gene_file,
    verbose,
):

    hierarchies_file_url = verify_source_of_data(
        hierarchy_file, url=HIERARCHIES_URL, allow_downloads=allow_downloads
    )

    pathway_to_gene_url = verify_source_of_data(
        pathway_to_gene_file, url=GENE_LIST_URL, allow_downloads=allow_downloads
    )
    pathway_to_gene_df = get_gene_list(pathway_to_gene_url)
    path_way_dict = pathway_to_gene_df["genes"].to_dict()
    hire_dict = get_hierarchy_data(hierarchies_file_url)
    path_2_gene = PathwaySeeks(path_way_dict, hire_dict)

    for pathway_idx in pathway_identifier:
        pathways = hire_dict[pathway_idx]
        outcomes = pathways_2_one_hot(pathways, path_2_gene)
        symbols = pd.Series(outcomes.index, name="symbol")
        dump_task_definitions(
            symbols, outcomes, main_task_directory, pathway_identifier
        )
        if verbose:
            print(
                f"{pathway_idx} was created at {main_task_directory} shaped {outcomes.shape}"
            )


if __name__ == "__main__":
    main()
