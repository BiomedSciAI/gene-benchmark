import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from gene_benchmark.task_retrieval import (
    list_form_to_onehot_form,
    verify_source_of_data,
)
from gene_benchmark.tasks import dump_task_definitions

TOP_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
GENE_LIST_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def get_gene_list(url):
    gene_list_file = pd.read_csv(url, on_bad_lines="skip", header=None, delimiter="\t")
    pathway_to_gene_df = pd.DataFrame(columns=['pathway_description', 'pathway', 'genes'])
    pathway_to_gene_df[['pathway_description', 'pathway']] = gene_list_file.iloc[:, :2]
    pathway_to_gene_df["genes"] = gene_list_file.iloc[:, 2:].apply(
        lambda x: [str(v) for v in set(x) if not pd.isna(v)], axis=1
    )
    pathway_to_gene_df = pathway_to_gene_df.set_index("pathway")
    pathway_to_gene = pathway_to_gene_df['genes'].to_dict()

    return pathway_to_gene

def get_hierarchy_data(url):
    hierarchy_df = pd.read_csv(url, header = None, delimiter="\t", names=['pathway', 'direct_child'])
    hierarchy_df = hierarchy_df.set_index("pathway")

    hierarchies = hierarchy_df.groupby(hierarchy_df.index)['direct_child'].apply(list).to_dict()
    return hierarchies

@dataclass
class PathwaySeeks:
    pathway_to_gene: Dict[str, List[str]]
    hierarchies: Dict[str, List[str]]
    missing_pathways_count: int = field(default=0, init=False)

    def populate_pathway(self, pathways: List[str], collected_paths: List[str] = None) -> List[str]:
        if collected_paths is None:
            collected_paths = []
        res = []
        for path in pathways:
            collected_paths.append(path)
            if path in self.pathway_to_gene:
                res.extend(self.pathway_to_gene[path])
            else:
                self.missing_pathways_count += 1
                print(f"Warning: Pathway {path} not found in pathway_to_gene.")
            if path in self.hierarchies:
                # recursively get genes from child pathways
                for child_path in self.hierarchies[path]:
                    child_genes, collected_paths = self.populate_pathway([child_path], collected_paths)
                    res.extend(child_genes)
        return list(set(res)), collected_paths
    
    def build_multilabel_task(self, pathway: str, result_genes: List[str], collected_paths: List[str]) -> pd.DataFrame:
        data = {path: [1 if gene in self.pathway_to_gene.get(path, []) else 0 for gene in result_genes] for path in collected_paths}
        return pd.DataFrame(data, index=result_genes)
    

# def get_top_level_pathway(hierarchies_df: pd.DataFrame) -> set[str]:
#     """
#     Returns the top level pathways from the table of pathways hierarchies.
#         top level are defined as pathways without a parent.

#     Args:
#     ----
#         hierarchies_df (pd.DataFrame): A data frame with a parent and child headers

#     Returns:
#     -------
#         set[str]: a set of top level pathways

#     """
#     pathway_that_are_parents = set(hierarchies_df["parent"].values)
#     pathway_that_are_children = set(hierarchies_df["child"].values)
#     pathway_who_are_just_parents = pathway_that_are_parents - pathway_that_are_children
#     return pathway_who_are_just_parents


# def create_top_level_task(
#     hierarchies_df: pd.DataFrame,
#     df_path: pd.DataFrame,
#     entities_name: str = "symbol",
#     pathway_names: str = "Pathway name",
# ) -> tuple[pd.Series, pd.DataFrame]:
#     """
#     Creates a top level tasks.

#     Args:
#     ----
#         hierarchies_df (pd.DataFrame): The pathways hierarchies table used to find the top pathways
#         df_path (pd.DataFrame): The pathways themselves, used to extract the gene list.
#         entities_name (str, optional): name of the entities. Defaults to 'symbol'.
#         pathway_names (str, optional): names of the pathways (converted from identifiers). Defaults to "Pathway name".

#     Returns:
#     -------
#         tuple[pd.Series,pd.DataFrame]: _description_

#     """
#     top_level = get_top_level_pathway(hierarchies_df)
#     top_in_file_paths = top_level.intersection(set(df_path.index))
#     df_path_top = df_path.loc[list(top_in_file_paths), :]
#     df_path_top.index = df_path_top[pathway_names]
#     outcomes = list_form_to_onehot_form(df_path_top)
#     symbols = pd.Series(outcomes.index, name=entities_name)
#     return symbols, outcomes


@click.command()
@click.option(
    "--pathway_symbol",
    type=click.STRING,
    help="Pathway from which we want to create dataframe for multilabel task.",
    default=None,
)
# @click.option(
#     "--main-task-directory",
#     "-m",
#     type=click.STRING,
#     help="The task root directory.  Will not be created.",
#     default="./tasks",
# )
# @click.option(
#     "--allow-downloads",
#     type=click.BOOL,
#     help="If false data files will be downloaded directly from HGNC and reactome, set to true only if you trust the urls above",
#     default=False,
# )
# @click.option(
#     "--pathways-file",
#     type=click.STRING,
#     help=f"A file with the pathway and the included symbol in {PATHWAYS_URL} format",
#     default=None,
# )
# @click.option(
#     "--pathways-relation-file",
#     type=click.STRING,
#     help="The location of the ReactomePathwaysRelation file available at https://reactome.org/download-data",
#     default=None,
# )
# @click.option(
#     "--verbose/--quite",
#     "-v/-q",
#     is_flag=True,
#     default=True,
# )
# @click.option(
#     "--add-top-pathways",
#     type=click.BOOL,
#     default=True,
# )
def main(
    # main_task_directory,
    # allow_downloads,
    # pathways_file,
    # pathways_relation_file,
    # verbose,
    # add_top_pathways
    pathway_symbol
    ):

    # pathways_relation_file = verify_source_of_data(
    #     pathways_relation_file, url=TOP_PATHWAYS_URL, allow_downloads=allow_downloads)

    pathway_to_gene = get_gene_list(GENE_LIST_URL)
    hierarchies = get_hierarchy_data(TOP_PATHWAYS_URL)

    pathway = PathwaySeeks(pathway_to_gene, hierarchies)
    result_genes, collected_paths = pathway.populate_pathway([pathway_symbol])
    print(f"Total missing pathways: {pathway.missing_pathways_count}")

    if result_genes:
        df = pathway.build_multilabel_task(pathway_symbol, result_genes, collected_paths)
        df.save_csv(f"multilabel_task_dataframe_{pathway_symbol}.csv", index = True)

    # if add_top_pathways:
    #     symbols, outcomes = create_top_level_task(hierarchies_df, pathways_df)
    #     dump_task_definitions(symbols, outcomes, main_task_directory, "top pathways")
    # if verbose:
    #     print(f"{5} was created at {main_task_directory} shaped {outcomes.shape}")
    # return


if __name__ == "__main__":
    main()
