# How to reproduce the manuscript
In the paper ["Does your model understand genes? A benchmark of gene properties for biological and text models"](https://arxiv.org/html/2412.04075) we applied the benchmark to a set of models from various sources. In this guide we will present how to duplicate the results.
The reproduction is computationally intensive so we separated it into several steps:
1. Embedding extraction
2. Running the tasks
3. Creation of figures

For simplicity in the paper we used a fix set of genes available [gene list](overlap_list.yaml). The scripts in this guide used the following environment variables:
```
export MODEL_FILES_FOLDER=""
export EMBEDDING_FOLDER=""
```

## Step 1: Embedding extraction
### How to extract textual embeddings
Here we extract the embeddings of the textual representations of the model (
    we recommend creating a fresh gene-benchmark virtual environment with `numpy 1.26`. We noted that with some `hugginface` models (specifically `DNABert2`) `numpy 1.26` works better). Following is the yaml file defining the model:
```
descriptor:
  class_name: NCBIDescriptor
encoder:
  class_name: SentenceTransformerEncoder
  class_args:
    encoder_model_name: "Salesforce/SFR-Embedding-Mistral"
model_name: MTEB-L
```

This is the file content for the MTEB-L ([`Salesforce/SFR-Embedding-Mistral`](https://huggingface.co/Salesforce/SFR-Embedding-Mistral)) In the manuscript we applied a similar file for two mode encoders: MTEB-S ([`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)) and MPNet ([`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).
After setting the Once the files are in place the following script should be applied:

```
python scripts/extract_gene_text_embeddings.py -e $MODEL_FILES_FOLDER/encoding_extraction/MTEB-S.yaml -e $MODEL_FILES_FOLDER/encoding_extraction/MTEB-L.yaml -e $MODEL_FILES_FOLDER/encoding_extraction/MPNet.yaml --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml
```


### How to extract base pair embeddings
Here we decided to separate the process into two parts because the textual extraction might a long time. so we are going to use the following model to save the descriptions in another .csv file:

```
descriptor:
  class_name: BasePairDescriptor
encoder:
  class_name: BERTEncoder
  class_args:
    encoder_model_name: "zhihan1996/DNABERT-2-117M"
    trust_remote_code: True
    tokenizer_name: "zhihan1996/DNABERT-2-117M"
model_name: DNABERT2
```

The extraction can use the following script:

```
python scripts/extract_gene_text_descriptions.py -e $MODEL_FILES_FOLDER/encoding_extraction/dnabert2.yaml --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml --allow-missing False
```

Now we will define an encoder model that uses the pre computed descriptions put the following in a DNABERT2_pre.yaml file:

```
descriptor:
  class_name: CSVDescriptions
  class_args:
    csv_file_path: "____/embeddings/descriptions_overlap_list_dnabert2.csv"
    index_col: 0
    description_col: "0"
encoder:
  class_name: BERTEncoder
  class_args:
    encoder_model_name: "zhihan1996/DNABERT-2-117M"
    trust_remote_code: True
    tokenizer_name: "zhihan1996/DNABERT-2-117M"
    maximal_context_size: 8192
model_name: DNABERT2
```

and now we can use the same script as before:
```
python scripts/extract_gene_text_embeddings.py -e $MODEL_FILES_FOLDER/encoding_extraction/DNABERT2_pre.yaml --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml
```

### How to extract Bag of words encodings
The following script creates embedding for the gene symbols using the descriptions from NCBI and the bag of words algorithm.

```
python scripts/encodings_retrieval/extract_bag_of_words_encodings.py  --output-file-dir $EMBEDDING_FOLDER  -l True
```

### How to extract ScGPT encodings

ScGPT encodings can be obtained using the following virtual environment,

```
conda create -n scgpt python=3.9 -y && conda activate scgpt\
pip install scgpt pandas click gdown IPython numpy==1.26
```

And then the following script
```
python scripts/encodings_retrieval/extract_scGPT_encodings.py -l True --output-file-dir $EMBEDDING_FOLDER
```
To reproduce the blood embedding use the following line:
```
python scripts/encodings_retrieval/extract_scGPT_encodings.py -l True --output-file-dir $EMBEDDING_FOLDER -m blood
```

note: In some devices an error with the OpenMP might rise (might be along the lines of OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program) if so try and install the 'nomkl' package.

### How to extract Gene2vec encodings
Gene2vec encodings can be obtained using the following command (the gene-benchmark env worked well for us):
```
python scripts/encodings_retrieval/extract_gene2vec_encodings.py -l True --output-file-dir $EMBEDDING_FOLDER
```

### How to extract CellPLM encodings
CellPLM encodings can be obtained using the following command, note that you need to download this [config.json]("https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h/ckpt/20230926_85M.config.json?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0) and this [best.ckpt](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h/ckpt/20230926_85M.best.ckpt?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0) files we suggest placing them in the MODEL_FILES_FOLDER folder.

files this dropbox folder
we recommend that you will create a virtual env for this model using the following commands:
```
conda create -n cellplm python=3.9 -y && conda activate cellplm\
pip install cellplm pytorch pandas click mygene pathvalidate requests
```
Following that we could run the script

```
python scripts/encodings_retrieval/extract_cellPLM_encodings.py  --output-file-dir $EMBEDDING_FOLDER --input-file-dir $MODEL_FILES_FOLDER
```

pip install pandas click mygene pathvalidate requests pyyaml
