import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from gene_benchmark.task_retrieval import (
    verify_source_of_data,
    list_form_to_onehot_form
)
from gene_benchmark.tasks import dump_task_definitions

HIERARCHIES_URL = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
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

    return pathway_to_gene, pathway_to_gene_df

def get_hierarchy_data(url):
    hierarchy_df = pd.read_csv(url, header = None, delimiter="\t", names=['parent', 'child'])
    hierarchy_df = hierarchy_df.set_index("parent")

    hierarchies = hierarchy_df.groupby(hierarchy_df.index)['child'].apply(list).to_dict()
    return hierarchies, hierarchy_df

@dataclass
class PathwaySeeks:
    pathway_to_gene: Dict[str, List[str]]
    pathway_to_gene_df: pd.DataFrame
    hierarchies: Dict[str, List[str]]
    missing_pathways_count: int = field(default=0, init=False)

    def populate_pathway(self, pathways: List[str]) -> (List[str], List[str]):
        collected_paths = []
        res = []
        for path in pathways:
            if path in self.pathway_to_gene:
                res.extend(self.pathway_to_gene[path])
            else:
                self.missing_pathways_count += 1
                print(f"Warning: Pathway {path} not found in pathway_to_gene.")
            if path in self.hierarchies:
                collected_paths.extend(self.hierarchies[path])
                # recursively get genes from child pathways
                for child_path in self.hierarchies[path]:
                    child_genes, _ = self.populate_pathway([child_path])
                    res.extend(child_genes)
        return list(set(res)), collected_paths
    
    def build_multilabel_task(self, pathway: str, result_genes: List[str], collected_paths: List[str]) -> pd.DataFrame:
        pathway_names = [self.pathway_to_gene_df.loc[path, 'pathway_description'] for path in collected_paths if path in self.pathway_to_gene_df.index]
        data = {
            "Submitted entities found": [";".join(sorted(set(self.pathway_to_gene.get(path, [])))) for path in collected_paths]
        }
        return pd.DataFrame(data, index=pathway_names)
    

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
    default=None,
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
        hierarchy_file, url=HIERARCHIES_URL, allow_downloads=allow_downloads)

    pathway_to_gene_url = verify_source_of_data(
        pathway_to_gene_file, url=GENE_LIST_URL, allow_downloads=allow_downloads)
    
    pathway_to_gene, pathway_to_gene_df = get_gene_list(pathway_to_gene_url)
    hierarchies, hierarchy_df = get_hierarchy_data(hierarchies_file_url)

    pathway = PathwaySeeks(pathway_to_gene, pathway_to_gene_df, hierarchies)
    result_genes, collected_paths = pathway.populate_pathway([pathway_identifier])
    print(f"Total missing pathways: {pathway.missing_pathways_count}")

    if result_genes:
        df_task= pathway.build_multilabel_task(pathway_identifier, result_genes, collected_paths)
        outcomes = list_form_to_onehot_form(df_task)
        symbols = pd.Series(outcomes.index, name='symbol')

        if pathway_identifier in pathway_to_gene_df.index:
            pathway_description = pathway_to_gene_df.loc[pathway_identifier, 'pathway_description']
        else:
            pathway_description = f"Pathway_{pathway_identifier}"

        dump_task_definitions(symbols, outcomes, main_task_directory, pathway_description)

        if verbose:
            print(
                f"{pathway_description} was created at {main_task_directory} shaped {outcomes.shape}"
            )

if __name__ == "__main__":
    main()
