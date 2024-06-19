from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from gene_benchmark.descriptor import (
    SingleEntityTypeDescriptor,
)
from gene_benchmark.encoder import (
    SentenceTransformerEncoder,
)


def convert_to_mat(data: pd.Series | pd.DataFrame):
    """
    Convert a 1d series or df with np arrays as values to a 2D/3D np array.

    Args:
    ----
        data: pd series or data frame

    returns: a converted np array matrix

    """
    if isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            return _convert_1d_df_to_2d_array(data)
        else:
            return _convert_df_to_3d_array(data)
    elif isinstance(data, pd.Series):
        return _convert_series_to_2d_array(data)
    else:
        raise TypeError("entities should be of type pandas DataFrame or pandas Series")


def _convert_series_to_2d_array(data: pd.Series):
    """
    Convert a 1d series or df with single columns with np arrays as values to a 2D np array.

    Args:
    ----
        data: pd series or data frame with single column

    returns: 2d np array

    """
    return np.vstack(data)


def _convert_1d_df_to_2d_array(data: pd.DataFrame):
    """
    Convert a 1d series or df with single columns with np arrays as values to a 2D np array.

    Args:
    ----
        data: pd series or data frame with single column

    returns: 2d np array

    """
    data = data.iloc[:, 0]
    return np.vstack(data)


def _convert_df_to_3d_array(df: pd.DataFrame):
    """
    Convert a multi dimensional dataframe with np arrays as values to a 3D np array.

    Args:
    ----
        df: pd data frame

    returns: 3d np array

    """
    n_rows, n_cols = df.shape
    n_embeding = len(df.iloc[0, 0])
    reshaped_embeddings = []
    for col in df.columns:
        reshaped_embeddings.append(
            np.stack(df[col].to_numpy()).reshape(n_rows, 1, n_embeding)
        )
    return np.hstack(reshaped_embeddings)


@dataclass
class TaskDefinition:
    """
    Class for keeping task definitions. Each class is composed of two elements. Entities (data frame) and outcomes (series).
    Entities is defined by a row of identifier, the type of identifier is the column names. For example a table with two gene columns
    |Gene |Gene   |
    |BRCA1|GATACA2|.

    """

    name: str
    entities: pd.DataFrame
    outcomes: pd.Series = None
    frac: int = 1
    sub_task: str | None = None

    def get_entities(self):
        return list(chain.from_iterable(self.entities.values))


def concat_list_df(list_df):
    if list_df.shape[1] > 1:
        list_df = list_df.apply(np.concatenate, axis=1)
    return np.stack(list_df)


class EntitiesTask:
    """
    A pipeline object that takes a task definitions object numerically encodes the entities
    using the encoder object and perform 5-fold cross validation prediction.
    (note that the scoring needs to fit the base_model).
    """

    def __init__(
        self,
        task: TaskDefinition | str,
        encoder: SentenceTransformer | str,
        tasks_folder: str | Path,
        description_builder: SingleEntityTypeDescriptor = None,
        base_model=LogisticRegression(max_iter=2000),
        cv: int = 5,
        scoring=("roc_auc",),
        return_estimator=True,
        encoding_post_processing: str = "average",
        exclude_symbols=None,
        frac=1,
        model_name=None,
        sub_task=None,
        multi_label_th=0,
        overlap_entities=False,
    ) -> None:
        """
        Initiate a ClassificationGeneTask object.

        Args:
        ----
            task (TaskDefinition,str): A task definition object or a task name
            encoder (Encoder): An encoder object
            base_prediction_model (sklearn classifier, optional): The prediction model to be used for the task. Defaults to LogisticRegression(max_iter=2000).
            cv (int, optional): The number of folds for the results. Defaults to 5.
            scoring (tuple, optional): the scoring tuple can use any method valid for sklearn's cross_validate . Defaults to ("roc_auc",).
            return_estimator (bool): If true the prediction model is saved in the object
            encoding_post_processing (str): The type of post processing for the encoding data frame either turning it into a 2D matrix by averaging
                                            (using "average") or by concatenation (selecting "concat") default is average
            exclude_symbols (list[str]|None): entries that contains any of these symbols will be excluded from the task
                This is used to compare different methods that may have a different set of covered symbols.  Defaults to None.
            frac (float): Randomly selects a unique fraction of the rows for the task default: 1
            sub_task(str|None): Use only one of the columns of the outcome as a binary task
            tasks_folder(str|None): Use an alternative task repository (default repository if None)

        """
        if isinstance(task, str):
            self.task_definitions = load_task_definition(
                task,
                tasks_folder=tasks_folder,
                exclude_symbols=exclude_symbols,
                frac=frac,
                sub_task=sub_task,
                multi_label_th=multi_label_th,
            )
        else:
            self.task_definitions = task
        if isinstance(encoder, str):
            self.encoder = SentenceTransformerEncoder(encoder)
        else:
            self.encoder = encoder
        self.exclude_symbols = exclude_symbols
        self.description_builder = description_builder
        self.base_prediction_model = base_model
        if type(cv) == int:
            self.cv = StratifiedKFold(n_splits=cv, shuffle=True)
        else:
            self.cv = cv
        self.scoring = scoring
        self._cv_report = None
        self.return_estimator = return_estimator
        self.encoding_post_processing = encoding_post_processing
        self.model_name = model_name
        self.overlap_entities = overlap_entities

    def _create_encoding(self):
        if self.description_builder is None:
            to_encode = self.task_definitions.entities.squeeze()
        else:
            to_encode = self.description_builder.describe(
                entities=self.task_definitions.entities
            )
        return to_encode

    def _post_processing_mat(self, encoding):
        if self.encoding_post_processing == "average":
            encodings_mat = convert_to_mat(encoding)
            if len(encodings_mat.shape) > 2:
                encodings_mat = np.average(encodings_mat, axis=1)
            return encodings_mat
        if self.encoding_post_processing == "concat":
            return concat_list_df(encoding)

    def _prepare_datamat_and_labels(self):
        descriptions_df = self._create_encoding()
        encodings_df = self.encoder.encode(descriptions_df)

        if self.overlap_entities:
            nan_ind = encodings_df.isna().any(axis=1)
            outcomes = self.task_definitions.outcomes.loc[~nan_ind]
            encodings = self._post_processing_mat(encodings_df.dropna())
            self.overlap_idx = nan_ind
        else:
            outcomes = self.task_definitions.outcomes
            encodings = self._post_processing_mat(encodings_df)
        self.overlapped_task_size = len(outcomes)
        return outcomes, encodings

    def run(self, error_score=np.nan):
        """
        Runs the defined ina k-fold fashion and returns a dictionary with the scores
        error_score: exposing a cross_validate option - on error in computatin:
            return np.nan (default) or error_score="raise" to raise an exception.
            Useful for debugging.  Default follows default of cross_validate function.

        Returns
        -------
            dict: a dictionary containing the score results and metadata on the run.

        """
        outcomes, encodings = self._prepare_datamat_and_labels()
        cs_val = cross_validate(
            self.base_prediction_model,
            encodings,
            outcomes,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=self.return_estimator,
            error_score=error_score,
            n_jobs=-1,
        )
        self._cv_report = cs_val
        return cs_val

    def summary(self):
        """
        Returns a dictionary with task parameters including the summaries of the description_builder and encoder.

        Returns
        -------
            dictionary: returns a dictionary with descriptions

        """
        summary_dict = {}
        summary_dict["task_name"] = self.task_definitions.name
        if self.task_definitions.sub_task is not None:
            summary_dict["sub_task"] = self.task_definitions.sub_task
        summary_dict["base_prediction_model"] = str(self.base_prediction_model)
        summary_dict["sample_size"] = self.task_definitions.outcomes.shape[0]
        is_bin = (
            self.task_definitions.outcomes.nunique() == 2
            if isinstance(self.task_definitions.outcomes, pd.Series)
            else False
        )
        if self.overlap_entities:
            summary_dict["overlapped_sample_size"] = len(self.nan_ind)
            outcomes = self.task_definitions.outcomes.loc[self.nan_ind]
        if is_bin:
            summary_dict["class_sizes"] = ",".join(
                [str(v) for v in outcomes.value_counts().values]
            )
            summary_dict["classes_names"] = ",".join(
                [str(v) for v in outcomes.value_counts().index.values]
            )
        if self.description_builder:
            summary_dict.update(self.description_builder.summary())
        summary_dict.update(self.encoder.summary())
        if self._cv_report:
            for scr in self.scoring:
                summary_dict[f"test_{scr}"] = ",".join(
                    [str(v) for v in self._cv_report[f"test_{scr}"]]
                )
                summary_dict[f"mean_{scr}"] = np.average(self._cv_report[f"test_{scr}"])
                summary_dict[f"sd_{scr}"] = np.std(self._cv_report[f"test_{scr}"])
        summary_dict["exclude_symbols_num"] = (
            len(self.exclude_symbols) if self.exclude_symbols else 0
        )
        summary_dict["model_name"] = self.model_name
        summary_dict["post_processing"] = self.encoding_post_processing
        summary_dict["frac"] = self.task_definitions.frac
        return summary_dict


def sub_sample_task_frames(entities, outcomes, frac=0.1):
    """
    Randomly select's a subset of rows. Selects the same rows from  entities and outcomes.

    Args:
    ----
        entities (pd.DataFrame): The entities data frame
        outcomes (pd.Series): The outcomes series
        frac (float, optional): The fraction of rows to select

    Returns:
    -------
        tuple (pd.DataFrame,pd.Series): A tuple of subselect entities and outcomes

    """
    base_ind = entities.sample(frac=frac).index
    if isinstance(outcomes, pd.DataFrame):
        return entities.loc[base_ind,], outcomes.loc[base_ind,]
    else:
        return entities.loc[base_ind,], outcomes[base_ind]


def filter_low_threshold_features(outcomes, threshold=0.1):
    """
    Filters out columns in the DataFrame where the mean of the column values is below a given threshold.

    Parameters
    ----------
    outcomes (pd.DataFrame): The input DataFrame containing feature columns.
    threshold (float): The threshold value below which columns will be removed.

    Returns
    -------
    pd.DataFrame: The filtered DataFrame with columns having a mean value above the threshold.

    """
    filtered_df = outcomes.loc[:, outcomes.mean() > threshold]
    return filtered_df


def load_task_definition(
    task_name: str,
    tasks_folder: str | Path,
    exclude_symbols=None,
    frac=1,
    sub_task=None,
    multi_label_th=0,
):
    """
    Loads and returns the task definition object.
    If exclude_symbols is given, all entries that have one of the symbols in one of the
    cells are filtered out, and the corresponding outcomes are filtered accordingly.


    Args:
    ----
        path (str): The folder containing two csv files one named entities and one named outcomes.
        exclude_symbols (list[str]|None): a list of symbols to exclude
        frac (float): load a unique fraction of the rows in the task, default 1
        tasks_folder(str|None): Use an alternative task repository (default repository if None)
        sub_task(str|None): Use only one of the columns of the outcome as a binary task


    Returns:
    -------
        task_definition: a task definition data class

    """
    _check_valid_task_name(task_name, tasks_folder=tasks_folder)
    if task_name == "Shared pathway":
        entities, outcomes = _load_gene_gene_pathway(
            task_name, tasks_folder, exclude_symbols
        )
    else:
        entities, outcomes = _load_task_definitions_from_folder(
            task_name, tasks_folder, exclude_symbols
        )

    if sub_task is not None:
        if not isinstance(outcomes, pd.DataFrame):
            raise ValueError(
                f"Task '{task_name}' has only one outcome and can not be used with sub_tasks"
            )
        if sub_task not in outcomes.columns:
            raise ValueError(
                f"Couldn't find the sub_task '{sub_task}'.  Known sub_tasks for task '{task_name}' tasks are {','.join(outcomes.columns)}"
            )
        outcomes = outcomes[sub_task].squeeze()

    if frac < 1:
        entities, outcomes = sub_sample_task_frames(entities, outcomes, frac=frac)

    if multi_label_th != 0:
        outcomes = filter_low_threshold_features(outcomes, threshold=multi_label_th)

    return TaskDefinition(
        name=task_name,
        entities=entities,
        outcomes=outcomes,
        frac=frac,
        sub_task=sub_task,
    )


def _load_task_definitions_from_folder(
    task_name: str,
    tasks_folder: str,
    exclude_symbols: list[str] | None = None,
    keep_default_na=False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads task definitions from a specific folder.
    If exclude_symbols is given, all entries that have one of the symbols in one of the
    cells are filtered out, and the corresponding outcomes are filtered accordingly.

    Args:
    ----
        task_name (str): the task name
        tasks_folder (str): the folder in which the text description is located
        excluded_symples (list[str]|None): a list of symbols to exclude
        keep_default_na(bool): "NA" in the input file is translated into NaN by pd,
            and NA can (and is) a gene name.  Turn on to read NaNs.  Treatmen of NaN values
            was not tested.

    Returns:
    -------
        entities(pd.DateFrame): table of entities for the tast
        outcomes(pd.Series): outcomes for the task

    """
    entities_file = get_lowest_entities_file(tasks_folder, task_name)
    if not entities_file:
        raise ValueError(
            f"could not find an entities file between {tasks_folder/task_name} and {tasks_folder}"
        )
    entities = pd.read_csv(
        tasks_folder / entities_file, keep_default_na=keep_default_na
    )
    outcomes = pd.read_csv(
        tasks_folder / task_name / "outcomes.csv", keep_default_na=keep_default_na
    ).squeeze()

    if exclude_symbols:
        entities, outcomes = filter_exclusion(entities, outcomes, exclude_symbols)
    return entities, outcomes


def get_lowest_entities_file(tasks_folder, task_name):
    path = Path(task_name)
    while path != Path("."):
        if (tasks_folder / path / "entities.csv").exists():
            return tasks_folder / path / "entities.csv"
        path = path.parent


def filter_exclusion(
    entities: pd.DataFrame, outcomes: pd.Series, excluded_symbols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    is_not_excluded_row = ~entities.map(lambda x: x in excluded_symbols).max(axis=1)
    return entities[is_not_excluded_row], outcomes[is_not_excluded_row]


def get_tasks_definition_names(tasks_folder: str | Path):
    """
    Get a list of tasks from the tasks folder.

    Args:
    ----
        tasks_folder(str) folder to check.  Defaults to regular task data folder

    Returns:
    -------
        list[str]: list of tasks

    """
    return [
        str(path.relative_to(tasks_folder))
        for path in Path(tasks_folder).rglob("*")
        if path.is_dir() and path.glob("*.csv")
    ]


def _check_valid_task_name(task_name: str, tasks_folder: Path):
    """
    Checks that there is a task names task_name in the designated folder.

    Args:
    ----
        task_name(str): the name of the task
        tasks_folder(str): data folder for the tasks

    Raises:
    ------
        ValueError: thrown if a folder for the task was not identified

    """
    known_tasks = get_tasks_definition_names(tasks_folder=tasks_folder)
    if not task_name in known_tasks:
        raise ValueError(
            f"Couldn't find the task {task_name} known tasks are {','.join(known_tasks)}"
        )
    return


def _load_gene_gene_pathway(
    task_name: str,
    tasks_folder: str,
    exclude_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a shared pathway task directly from the pathways description file creating the entities and outcomes
    during runtime.

    Args:
    ----
        task_name(str): The task name containing the pathway description file
        tasks_folder(str): The folder containing the task
        exclude_symbols (list[str]): List of symbols to be excluded.

    Returns:
    -------
        entities(pd.DateFrame): table of entities for the tast
        outcomes(pd.Series): outcomes for the task
    ------

    """
    path_ways = pd.read_csv(Path(tasks_folder) / Path(task_name) / "top_paths.csv")
    path_list = (
        path_ways["Submitted entities found"].apply(lambda x: set(x.split(";"))).values
    )
    any_pathway_genes = chain.from_iterable(path_list)
    gene_tuple = list(combinations(any_pathway_genes, 2))
    entities = pd.DataFrame(data=gene_tuple, columns=["symbol", "symbol"])
    outcomes = entities.apply(
        lambda r: _share_path(path_list, r.values[1], r.values[0]), axis=1
    )
    if exclude_symbols:
        entities, outcomes = filter_exclusion(entities, outcomes, exclude_symbols)
    return entities, outcomes


def _share_path(path_list, gene_a, gene_b):
    """
    Checks if gene a and gene b appear together in on the paths.
    We stop at the first shared path to slightly improve run time.

    Args:
    ----
        path_list[set]: A list of sets. Each set represent a pathway
        gene_a(str): a gene name
        gene_b(str): a gene name.

    Returns:
    -------
        Bool: True if both genes share a path i.e. a cell in the list
    ------

    """
    for path in path_list:
        if gene_a in path:
            if gene_b in path:
                return True
    return False
