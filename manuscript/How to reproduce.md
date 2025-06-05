# How to reproduce the manuscript

Below are instructions to reproduce the benchmark as executed in the paper ["Does your model understand genes? A benchmark of gene properties for biological and text models"](https://arxiv.org/html/2412.04075).
## Benchmark flow

1. Obtain embeddings from a biological or text model. 
2. Run tasks on embeddings based on the tasks defined in [tasks], which can be populated by running the task download scripts [here]()

These steps can be run all at once, for the text models, though it is recommended to save the embeddings to csv in a separate step because they may be slow.

The main script for reproducing the benchmark is `run_task`. It expects a yaml with fields `descriptor` and `decoder`. The `descriptor` translates the gene symbol into a gene description based on its DNA sequence for the DNA models, or NCBI description for the text models.
The Encoder describes the foundation model to use for encoding the gene symbols to vectors.

Note: the embeddings that are extracted from the biological models are not part of the same flow. They must be prepared in advance, as described below in the section [Embeddings](#Embeddings). Once prepared, they can be accessed as `PreComputedEncoder` encoders as shown in the sample files.

## How to reproduce the performance tables
The following lines scripts will retrieve the embeddings per model, preform the tasks and save the results. Note that for Gene2Vec, Geneformer, ScGPT blood\human, CellPLM, DNABERT2, ESM2 and Bag of Words models we need to extract the embeddings and set the files to their locations (How to extract the embeddings follows).

```
# run the binary tasks
python scripts/run_task.py -t scripts/task_configs/binary_tasks.yaml --output-file-name ~/reports/binary.csv -i /manuscript/gene_set.yaml -m ~/manuscript/models/BGW.yaml -m /manuscript/models/cellPLM_pre.yaml -m /manuscript/models/esm2_pre.yaml -m /manuscript/models/Gene2Vec_pre.yaml -m /manuscript/models/Geneformer.yaml -m /manuscript/models/MPNet.yaml -m /manuscript/models/MTEB-L.yaml -m /manuscript/models/MTEB-S.yaml -m /manuscript/models/ScGPT-H.yaml -m /manuscript/models/ScGPT-B.yaml

python scripts/run_task.py -t scripts/task_configs/categorical_tasks.yaml --output-file-name ~/reports/multi.csv -i /manuscript/gene_set.yaml -m ~/manuscript/models/BGW.yaml -m /manuscript/models/cellPLM_pre.yaml -m /manuscript/models/esm2_pre.yaml -m /manuscript/models/Gene2Vec_pre.yaml -m /manuscript/models/Geneformer.yaml -m /manuscript/models/MPNet.yaml -m /manuscript/models/MTEB-L.yaml -m /manuscript/models/MTEB-S.yaml -m /manuscript/models/ScGPT-H.yaml -m /manuscript/models/ScGPT-B.yaml -s category

python scripts/run_task.py -t scripts/task_configs/multi_label_tasks.yaml --output-file-name ~/reports/multi.csv -i /manuscript/gene_set.yaml -m ~/manuscript/models/BGW.yaml -m /manuscript/models/cellPLM_pre.yaml -m /manuscript/models/esm2_pre.yaml -m /manuscript/models/Gene2Vec_pre.yaml -m /manuscript/models/Geneformer.yaml -m /manuscript/models/MPNet.yaml -m /manuscript/models/MTEB-L.yaml -m /manuscript/models/MTEB-S.yaml -m /manuscript/models/ScGPT-H.yaml -m /manuscript/models/ScGPT-B.yaml  -s multi --multi-label-th 0.05

python scripts/run_task.py -t scripts/task_configs/derived_binary_tasks.yaml --output-file-name ~/reports/multi.csv -i /manuscript/gene_set.yaml -m ~/manuscript/models/BGW.yaml -m /manuscript/models/cellPLM_pre.yaml -m /manuscript/models/esm2_pre.yaml -m /manuscript/models/Gene2Vec_pre.yaml -m /manuscript/models/Geneformer.yaml -m /manuscript/models/MPNet.yaml -m /manuscript/models/MTEB-L.yaml -m /manuscript/models/MTEB-S.yaml -m /manuscript/models/ScGPT-H.yaml -m /manuscript/models/ScGPT-B.yaml
```

## Saving gene embeddings to use in `gene-benchmark`
The biological models and the bag of words models require to save the embeddings as a `.csv` files before hand this can easily done by the following:


### Bag of words
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

 ### ESM2
 for instruction on how to extract ESM2 embeddings see this [link](https://github.com/snap-stanford/SATURN/blob/main/protein_embeddings/Generate%20Protein%20Embeddings.ipynb)


## Advanced topics
### How to extract text models embeddings encodings
To improve run time of the text models we can also save their embeddings using the following script.

```
python scripts/extract_gene_text_embeddings.py -e /manuscript/models/MTEB-S.yaml -e /manuscript/models/MTEB-L.yaml -e /manuscript/models/MPNet.yaml --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml
```
Once extracted the user can define a precomputed encoder using those files as well.


### How to extract base pair descriptions
Downloading the base pair descriptions can be time consuming to optimize this process the user can download the description using the following script

```
python scripts/extract_gene_text_descriptions.py -e /manuscript/models/dnabert2.yaml --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml --allow-missing False
```
Following the user can use the downloaded descriptions for the DNABert model see the
Now we will define an encoder model that uses the pre computed descriptions put the following in a 'DNABERT2_pre.yaml' model file. This can be used to save the embeddings as well just like the text models.
```
python scripts/extract_gene_text_embeddings.py -e /manuscript/models/DNABERT2_pre.yaml  --output-folder $EMBEDDING_FOLDER --gene-symbols $MODEL_FILES_FOLDER/overlap_list.yaml
```
