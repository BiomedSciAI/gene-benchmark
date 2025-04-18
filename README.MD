# Gene Benchmark

A package to benchmark pre-trained models on downstream gene-related tasks.

The package adopts a framework that allows comparison of models using multiple input types by focusing on the learned embeddings the model provides for each entity. This allows comparing models trained on text, proteins, transcriptomics, or any other modality.

To learn more about the benchmark, check out the preprint "Does your model understand genes? A benchmark of gene properties for biological and text models" at <https://arxiv.org/abs/2412.04075> .

The repository is divided into the following sections:

* [gene_benchmark](./gene_benchmark/): The package itself, containing the scripts for extracting textual descriptions from NCBI, encoding textual descriptions and evaluation of model performance.

* [notebooks](./notebooks/): Notebooks for creating the results figures and package usage examples.

* [scripts](./scripts/): Scripts for description extraction, encoding, task creation and execution.

* [tasks](./tasks/): The default directory that will be populated with all the tasks after running the task creation script.

In depth explanation on each of the packages components can be found in the `gene_benchmark` directory.

## Environment

Using a virtual environment for all commands in this guide is strongly recommended.
The package works with `conda`, `uv` and vanilla `venv` environments.

```sh
# create a conda enviornment "gene_benchmark" with Python version 3.11
conda create -n gene_benchmark python=3.11

# activate the enviornment before installing new packages
conda activate gene_benchmark
```

## Installation

The package is not yet available on pypi. It can be installed directly from GitHub
The following command will install the repository as a Python package, and also attempt to install dependencies specified in the setup.py file or the pyproject.toml.

```sh
# Note that the command does not clone the repository.
pip install "git+https://github.com/BiomedSciAI/gene-benchmark.git"
```

or the repo can be cloned and installed manually.

```sh
git clone https://github.com/BiomedSciAI/gene-benchmark.git
# Change directory to the root of the cloned repository
cd gene-benchmark

# install from the local directory
pip install -e .
```

## Usage

To evaluate your model on the tasks a few basic steps need to be done:

### Set up

1. Create the tasks: This repo does not contain gene task data. Instead, we provide scripts to download and populate the tasks from diverse sources. To download the files containing the tasks, run the following commands in your terminal from the root directory. Note that each task dataset has its own license, which is separate from the license of this package.

```sh
python scripts/tasks_retrieval/gene2gene_task_creation.py --allow-downloads True
python scripts/tasks_retrieval/Genecorpus_tasks_creation.py --allow-downloads True
python scripts/tasks_retrieval/HLA_task_creation.py --allow-downloads True
python scripts/tasks_retrieval/HPA_tasks_creation.py --allow-downloads True
python scripts/tasks_retrieval/humantfs_task_creation.py --allow-downloads True
python scripts/tasks_retrieval/Reactome_tasks_creation.py  --allow-downloads True
python scripts/tasks_retrieval/uniprot_keyword_tasks_creation.py  --allow-downloads True
```

Now your [tasks](./tasks/) directory should be populated with subdirectories with the tasks names. Each subdirectory holds two .csv files, one with the gene symbols (`entities.csv`) and one with the labels (`outcomes.csv`). The shape of these csv files will be determined based on the task type. For example, for the multi class tasks, the outcomes will be a 2d matrix.

2. Create your task yaml: The script for running the tasks can receive either the task names themselves or a .yaml file containing the list of task names you wish to run. If you choose to create a .yaml file with the task names, create a separate file for each task type. For example for the binary tasks:

```sh
- TF vs non-TF
- long vs short range TF
- bivalent vs non-methylated
- Lys4-only-methylated vs non-methylated
- dosage sensitive vs insensitive TF
- Gene2Gene
- CCD Transcript
- CCD Protein
- N1 network
- N1 targets
- HLA class I vs class II
```

The example task configs can be found in [task_configs](./scripts/task_configs/)

3. Create the model config file: This config file will hold the path to your models embeddings and the name you wish to use for your model. The structure of the file:

```sh
encoder:
  class_name: PreComputedEncoder
  class_args:
    encoder_model_name: "/path/to/your/embeddings/my_models_embeddings.csv"
model_name: my_model_name
```

Note that the script expects the embedding csv file to have a 'symbol' column with the gene symbols, this will be set as the index.

### Run task

Each task type (binary, categorical or multi-label) will be run separately.
For example, for running the binary tasks the command is:

```sh
python scripts/run_task.py -t /path/to/task/yaml/base_binary.yaml -tf /tasks -m /path/to/model/config/model.yaml --output-file-name binary_tasks.csv
```

Note:

* For the other task types (categorical, regression or multi labe) you need to add `-s category/regression/multi`

* When you are running the tasks on multiple models, and you would like them to be comparable, you can add a `excluded-symbols-file` input. This needs to be a path to a yaml file containing a list of gene names you would like to exclude.

* To avoid getting errors during the cross validation due to class imbalance, you can add a threshold for the classes `"-th"` (for multi label)or `"-cth"` (for categorical)

## Citation

If you make use of this package, please cite our manuscript:

```tex
@misc{kantor2024doesmodelunderstandgenes,
      title={Does your model understand genes? A benchmark of gene properties for biological and text models},
      author={Yoav Kan-Tor and Michael Morris Danziger and Eden Zohar and Matan Ninio and Yishai Shimoni},
      year={2024},
      eprint={2412.04075},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.04075},
}
```
