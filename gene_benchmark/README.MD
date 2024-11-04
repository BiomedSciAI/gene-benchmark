# gene-benchmark

A benchmark for gene embeddings from pretrained models.

The package includes three main sections:

1. Descriptions - Retrieval of textual descriptions from pre existing files or database.
2. Encode  the textual data
3. Tasks - Evaluation of the efficacy of these encodings in downstream tasks.

## Description

Retrieval of textual descriptions from pre existing files or NCBI. Currently, the package can extract description for a single entity type (for example, gene symbols ..) or for multiple entity types (for example, gene symbols and disease...)

### Single entity type descriptions

Currently, the descriptions can be retrieved from the NCBI database and from a pre existing CSV with the text descriptions.

1. ```NCBIDescriptor```:
    Creates descriptions for gene symbol based on data from the NCBI. The descriptions are build as follows:
    "Gene symbol {symbol} full name {name} with the summary {summary}.
    The user can also decide if he would like partial descriptions (missing gene name or summary), if not, None will be returned for that gene.
2. ```CSVDescriptions```:
    Retrieves any entity description (gene, disease...) from a CSV file. The csv file needs to include a column with the entity names (default is 'id').

### Multi entity type descriptions

```MultiEntityTypeDescriptor```: This class needs a ```description_dict``` with the keys as the entity type and the value the wanted Single entity type descriptions class.

### Generating descriptions

After creating an instance of your desired class, use the method '''describe''' to generate the descriptions. For all classes, input a pandas series or dataframe with the entity names and set '''allow_missing''' to True if you don't mind to have missing entity descriptions, or to false if you would like the extraction the stop if some entities are missing.
The result will be the corresponding dataframe/series with the descriptions replacing the entities imputed.

### Summary

All methods have a summary attribute that will print a short summary of the description retrieval, it includes:

- If partial description were allowed
- If missing entities were allowed
- Number of missing entities
- List of missing entities

### Examples

#### extraction of single entity type descriptions from NCBI

```python
descriptor = NCBIDescriptor(allow_partial=False)
descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol"], index=[1, 2, 3], data=["BRCA1", "FOXP2", "BRCA1"]
            )
        )
```

#### extraction of single entity type descriptions from a CSV

```python
descriptor = CSVDescriptions(csv_file_path=csv_path, index_col="id")
descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"]), allow_missing=True
            )
```

#### extraction of multi entity types descriptions

```python
description_dict = {
            "symbol": NCBIDescriptor(allow_partial=False),
            "disease": CSVDescriptions(csv_file_path=csv_path, index_col="id"),
        }
descriptor = MultiEntityTypeDescriptor(description_dict=description_dict)
descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol", "disease", "symbol"],
                data=[("BRCA1", "cancer", "FOXP2"), ("PLAC4", "als", "IAMNOTAGENE")],
            )
        )
```

## Encoding

### PreComputedEncoder

This class enable to work with dataframe that contain index for the encoded elements (usually symbols) and the encoding themselves as the columns.
The following is a way to load the Gene2Vec csv of encoding as an encoder:

```python
enc = PreComputedEncoder(encoder_model_name: "/path/to/gene2vec.csv")
```

The precomputed encoder actually maps specific strings to specific encoding. Hence it is limited in it's encoding scope and usually will work with gene symbols or disease id's .

### SentenceTransformerEncoder

In addition we enable encoding using the HuggingFace sentence encoders [see](https://huggingface.co/sentence-transformers)

```python
enc = SentenceTransformerEncoder(encoder_model_name: "BAAI/bge-large-en-v1.5")
```

Note that the models are downloaded and might need a lot of storage space.
You can make use of the environment variable `SENTENCE_TRANSFORMERS_HOME` to control where the models are downloaded to.
SentenceTransformerEncoder can encode any string and is not limited and is very useful when coupled with descriptions.

## Tasks

The package has the means to create, load and manipulate task definitions.
The list of available tasks is provided [here]( /tasks/task_descriptions.xlsx)
For each task we provided python scripts that enable to create the tasks either from a local file
or to download them directly.

### Creation of the tasks


#### How to load tasks definitions

One can easily load the task definitions using the `load_task_definition` method and the task name (according to the list above)

```python
from gene_benchmark.tasks import TaskDefinition
tasks = load_task_definition(task_name='TF vs non-TF')
```

If the user wants to use a task but wishes to exclude certain symbols he can easily do so using `exclude_symbols`

```python
tasks = load_task_definition(task_name='TF vs non-TF',exclude_symbols=['BRCA1'])
```

If the user wants to use one of the labels of a multi-label task, it can be done by:
```python
tasks = load_task_definition(task_name='Pathways',sub_task='Diseases')
```

Or the user wishes to load a multi-label task but only the labels with a certain label rate
```python
tasks = load_task_definition(task_name='Pathways',multi_label_th=0.1)
```

When running tasks that are not inside the main task repository, they can be loaded from a different data dir by setting 'data_dir' to the alternative task directory.

#### How to run a task

The package includes a pipeline object that given a task, prompt maker and encoder can preform the entire task from load to predictions.

```python
from gene_benchmark.tasks import EntitiesTask
from gene_benchmark.descriptor import NCBIDescriptor
from gene_benchmark.encoder import SentenceTransformerEncoder,


task = EntitiesTask(task_name='TF vs non-TF', encoder=SentenceTransformerEncoder(), prompt_builder=NCBIPromptsMaker())
_ = task.run()
print(task.summary())
```

### The `run_task.py` script

The script enables the performance of multiple tasks on multiple models.
Each model is defined by an encoder and an prompt maker. The script uses yaml files to define each model. The model_name field is used for the report

#### NCBI gene descriptions

Following is a model that uses NCBI prompts and hugginface sentence encoder.

```yaml
descriptor:
  class_name: NCBIPromptsMaker
encoder:
  class_name: SentenceTransformerEncoder
  class_args:
    encoder: "sentence-transformers/all-mpnet-base-v2"
model_name: mpnet
```

#### gene symbols

We can create a model that encodes only the symbols:

```yaml
encoder:
  class_name: SentenceTransformerEncoder
  class_args:
    encoder: "sentence-transformers/all-mpnet-base-v2"
```

#### Gene2Vec precomputed encoders

We also support Gene2Vec using the precomputed encoders:

```yaml
encoder:
  class_name: PreComputedEncoder
  class_args:
    encoder_model_name: "/path/to/gene2vec.csv"
```

#### scGPT precomputed encoding

See [scGPT](https://github.com/bowang-lab/scGPT)
The encoding was extracted from the pre-trained "blood" model.
Weights file can be generated using the [extraction script](scripts/data_retrieval/scGPT_embedding_extractor.py)

```yaml
encoder:
  class_name: PreComputedEncoder
  class_args:
    encoder_model_name: "/path/to/ScGPT_weights/blood_model_embedding.csv"
```
