import os
from os.path import isfile
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from yaml import safe_load

from gene_benchmark.deserialization import load_class
from gene_benchmark.tasks import EntitiesTask


def get_report(task, output_file_name, append_results):
    """
    Create a summary dataframe, if append_results it loads the output_file_name and append the summary from res.

    Args:
    ----
        task : A task object with the data to summarize
        output_file_name (str): the file to append to
        append_results (bool): whether or not to append

    Returns:
    -------
        pd.DataFrame: A summary data frame

    """
    this_run_df = pd.DataFrame.from_dict(task.summary(), orient="index")
    this_run_df = this_run_df.transpose()
    if append_results and isfile(output_file_name):
        old_report = pd.read_csv(output_file_name)
        this_run_df = pd.concat([old_report, this_run_df], axis=0)
    return this_run_df


def load_yaml_file(model_config):
    with open(model_config) as f:
        loaded_yaml = safe_load(f)
    return loaded_yaml


def expand_task_list(task_list):
    tsk_list = []
    for task in task_list:
        if ".yaml" in task:
            tsk_list = tsk_list + load_yaml_file(task)
        else:
            tsk_list.append(task)
    return tsk_list


@click.command()
@click.option(
    "--tasks-folder",
    "-tf",
    type=click.STRING,
    help="The folder where tasks are stored. Defaults to `GENE_BENCHMARK_TASKS_FOLDER`",
    default=None,
)
@click.option(
    "--task-names",
    "-t",
    type=click.STRING,
    help="The output file name.",
    default=["long vs short range TF"],
    multiple=True,
)
@click.option(
    "--model-config-files",
    "-m",
    type=click.STRING,
    help="Append results to the files",
    default=[str(Path(__file__).parent / "models" / "ncbi_multi_class.yaml")],
    multiple=True,
)
@click.option(
    "--excluded-symbols-file",
    "-e",
    type=click.STRING,
    help="A path to a yaml file containing symbols to be excluded",
    default=None,
)
@click.option(
    "--output-file-name",
    type=click.STRING,
    help="The output file name.",
    default="task_report.csv",
)
@click.option(
    "--append-results",
    type=click.BOOL,
    help="Append results to the files",
    default=True,
)
@click.option(
    "--verbose",
    type=click.BOOL,
    help="print progress",
    default=True,
)
@click.option(
    "--sub-sample",
    type=click.FLOAT,
    help="sub sample the task",
    default=1,
)
@click.option(
    "--scoring_type",
    "-s",
    type=click.STRING,
    help="use different scoring",
    default="binary",
)
@click.option(
    "--multi-label-th",
    "-th",
    type=click.FLOAT,
    help="threshold of imbalance of labels in multi class tasks",
    default=0.0,
)
def main(
    tasks_folder,
    task_names,
    model_config_files,
    excluded_symbols_file,
    output_file_name,
    append_results,
    verbose,
    sub_sample,
    scoring_type,
    multi_label_th,
):
    if tasks_folder is None:
        tasks_folder = Path(os.environ["GENE_BENCHMARK_TASKS_FOLDER"])
        assert tasks_folder.exists()
    if excluded_symbols_file:
        with open(excluded_symbols_file) as f:
            exclude_symbols = safe_load(f)
    else:
        exclude_symbols = []
    task_names = expand_task_list(task_names)
    for model_config in model_config_files:
        for task_name in task_names:
            if verbose:
                print("Started", model_config, task_name)
            with open(model_config) as f:
                model_dict = safe_load(f)
            if "descriptor" in model_dict:
                description_builder = load_class(**model_dict["descriptor"])
            else:
                description_builder = None
            if "base_model" in model_dict:
                base_model = load_class(**model_dict["base_model"])
            else:
                base_model = LogisticRegression(max_iter=5000, n_jobs=-1)
            if scoring_type == "category":
                scoring = (
                    "roc_auc_ovr_weighted",
                    "accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                )
                cv = 5
            elif scoring_type == "regression":
                scoring = (
                    "r2",
                    "neg_root_mean_squared_error",
                    "neg_mean_absolute_error",
                )
                cv = KFold(n_splits=5, shuffle=True)
                base_model = LinearRegression(n_jobs=-1)
            elif scoring_type == "multi":
                scoring = {
                    "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
                    "hamming_loss": make_scorer(hamming_loss),
                    "accuracy": make_scorer(accuracy_score),
                    "precision_weighted": make_scorer(
                        precision_score, average="weighted"
                    ),
                    "recall_weighted": make_scorer(recall_score, average="weighted"),
                    "f1_weighted": make_scorer(f1_score, average="weighted"),
                }
                cv = KFold(n_splits=5, shuffle=True)
                base_model = MultiOutputClassifier(
                    LogisticRegression(max_iter=5000, n_jobs=-1)
                )
            else:
                scoring = (
                    "roc_auc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                )
                cv = 5
                base_model = LogisticRegression(max_iter=5000, n_jobs=-1)
            if "model_name" in model_dict:
                model_name = model_dict["model_name"]
            else:
                model_name = None
            if "post_processing" in model_dict:
                post_processing = model_dict["post_processing"]
            else:
                post_processing = "average"
            encoder = load_class(**model_dict["encoder"])
            if ";" in task_name:
                sub_task = task_name.split(";")[1]
                task_name = task_name.split(";")[0]
            else:
                sub_task = None
            if verbose:
                print("Loaded", model_config, task_name, sub_task)
            task = EntitiesTask(
                task_name,
                tasks_folder=tasks_folder,
                encoder=encoder,
                description_builder=description_builder,
                exclude_symbols=exclude_symbols,
                scoring=scoring,
                base_model=base_model,
                cv=cv,
                frac=float(sub_sample),
                model_name=model_name,
                encoding_post_processing=post_processing,
                sub_task=sub_task,
                multi_label_th=multi_label_th,
            )
            _ = task.run()
            report_df = get_report(task, output_file_name, append_results)
            report_df.to_csv(output_file_name, index=False)
            print("Succeeded", model_config, task_name, sub_task)
    return


if __name__ == "__main__":
    main()
