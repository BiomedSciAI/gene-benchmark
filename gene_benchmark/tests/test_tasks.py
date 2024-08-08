import os
import random
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from gene_benchmark.descriptor import (
    NCBIDescriptor,
)
from gene_benchmark.encoder import (
    PreComputedEncoder,
)
from gene_benchmark.tasks import (
    EntitiesTask,
    TaskDefinition,
    _load_task_definitions_from_folder,
    convert_to_mat,
    dump_task_definitions,
    filter_exclusion,
    get_tasks_definition_names,
    load_task_definition,
    sub_sample_task_frames,
)


def _get_resources():
    return Path(__file__).parent / "resources"


def _get_test_tasks_folder():
    return _get_resources() / "tasks"


def _get_tasks_folder():
    git_location = Path(__file__).parents[2] / "tasks"
    tasks_folder = Path(os.environ.get("GENE_BENCHMARKS_TASKS_FOLDER", git_location))
    assert tasks_folder.exists
    return tasks_folder


def _generate_class_task_definitions(
    sample_num=100,
    class_num=2,
    entities_type_num=1,
    outcome_num=1,
    entities_type_name=None,
    numeric_class=False,
    p=0.03,
):
    """Generates task definitions according to user specifications.

    Args:
        sample_num (int, optional): The number of samples in that task i.e. length. Defaults to 100.
        class_num (int, optional): the number of classes in the outcome. Defaults to 5.
        entities_type_num (int, optional): The number of entities type. Defaults to 1.
        outcome_num (int, optional): The number of outcomes. Defaults to 1.
        entities_type_name (list[str], optional): The names of each entities type. Defaults to None.
        numeric_class (bool, optional): If True the classes are numeric. Defaults to False.
        p (float, 0.03):

    Returns:
        tuple(pd.DataFrame,pd.DtaFrame|pd.Series): the entities and outcomes of a task
    """
    cls_names = {i: i if numeric_class else f"class_{i}" for i in range(class_num)}
    entities_list = [f"Gene_{i}" for i in range(sample_num)]
    data = np.vstack(
        [
            np.random.choice(entities_list, size=sample_num, replace=False)
            for i in range(entities_type_num)
        ]
    ).T
    entities_names = (
        ["symbol"] * entities_type_num
        if entities_type_name is None
        else entities_type_name
    )
    entities = pd.DataFrame(data=data, columns=entities_names)
    out_labels = [cls_names[i % class_num] for i in range(sample_num)]
    out_data = np.vstack(
        [
            np.random.choice(out_labels, size=sample_num, replace=False)
            for i in range(outcome_num)
        ]
    ).T
    out_columns = (
        ["Outcomes"]
        if outcome_num == 1
        else [f"Outcomes_{i}" for i in range(outcome_num)]
    )
    outcomes = pd.DataFrame(data=out_data, columns=out_columns).squeeze()
    if outcome_num >= 2 and class_num == 2 and numeric_class:
        outcomes.iloc[:, -1] = np.random.choice([0, 1], 100, p=[1 - p, p])
    return entities, outcomes


def _add_computed_tasks(main_task_directory):
    """Populate the test task folder with test. The method was used an the output is in the appropriate locations.

    Args:
        main_task_directory (str): The folder to populate with the testing tasks
    """
    task_dict = {
        "symbol_bin": {},
        "simple_cat": {"class_num": 5},
        "interaction": {"entities_type_num": 2},
        "two_int": {"entities_type_num": 2, "entities_type_name": ["symb1", "symb2"]},
        "multi_label": {"numeric_class": True, "outcome_num": 5, "class_num": 2},
    }
    for task_name, params in task_dict.items():
        dump_task_definitions(
            *_generate_class_task_definitions(**params), main_task_directory, task_name
        )


class TestTasks(unittest.TestCase):
    def _test_filter_entities(self, task_name):
        tasks_folder = _get_test_tasks_folder()
        entities, outcomes = _load_task_definitions_from_folder(task_name, tasks_folder)

        original_length = entities.shape[0]
        excluded = list(
            entities[entities.columns[0]][random.sample(range(original_length), 10)]
        )
        excluded += ["There is not gene called this way"]

        filtered_entities, filtered_outcome = filter_exclusion(
            entities, outcomes, excluded
        )
        assert filtered_entities.shape[0] == filtered_outcome.shape[0]
        assert filtered_entities.shape[0] < entities.shape[0]
        assert filtered_entities.shape[1] == entities.shape[1]
        # check that the excluded entities do not appear in any of the columns
        assert all(
            set(filtered_entities[col].values).isdisjoint(set(excluded))
            for col in filtered_entities
        )

    # test filter_exclusion function directly

    def test_filter_exclusion_directly_one_column(self):
        task_name = "symbol_bin"
        self._test_filter_entities(task_name)

    def test_filter_exclusion_directly_two_columns(self):
        task_name = "interaction"
        self._test_filter_entities(task_name)

    # test load_task_definition with exclusion

    def test_load_task_with_exclusion_one_column(self):
        task_name = "symbol_bin"
        self._test_load_task_with_exclusion(task_name)

    # test load from subdir
    def test_load_task_from_subdir(self):
        task_name = "test_multiclass/level1/level2/level3_with_data_local"
        self._test_load_task_without_exclusion(
            task_name, tasks_folder=_get_test_tasks_folder()
        )

    def test_load_task_from_subdir_when_entities_in_parent(self):
        task_name = "test_multiclass/level1/level2/subtest_with_data_above"
        self._test_load_task_without_exclusion(
            task_name, tasks_folder=_get_test_tasks_folder()
        )

    def test_load_task_from_subdir_when_entities_in_parent_bad(self):
        task_name = "a/b/c/d/e"
        with pytest.raises(ValueError, match="could not find an entities file"):
            self._test_load_task_without_exclusion(task_name)

    def test_load_task_with_exclusion_two_columns(self):
        task_name = "interaction"
        self._test_load_task_with_exclusion(task_name)

    # test load_task_definition with empty exclusion

    def test_load_task_without_exclusion_one_column(self):
        task_name = "symbol_bin"
        self._test_load_task_without_exclusion(task_name)

    def test_load_task_without_exclusion_two_columns(self):
        task_name = "interaction"
        self._test_load_task_without_exclusion(task_name)

    def test_load_multiclass_task_without_exclusion_one_column(self):
        task_name = "test_multiclass"
        self._test_load_task_without_exclusion(
            task_name, tasks_folder=_get_test_tasks_folder()
        )

    def _test_load_task_with_exclusion(self, task_name):
        tasks_folder = _get_test_tasks_folder()
        entities, outcomes = _load_task_definitions_from_folder(task_name, tasks_folder)

        original_length = entities.shape[0]
        excluded = list(
            entities[entities.columns[0]][random.sample(range(original_length), 10)]
        )

        task = load_task_definition(
            task_name=task_name, tasks_folder=tasks_folder, exclude_symbols=excluded
        )

        filtered_entities, filtered_outcome = filter_exclusion(
            entities, outcomes, excluded
        )
        assert filtered_entities.shape == task.entities.shape
        assert filtered_outcome.shape == task.outcomes.shape

        # check that the excluded entities do not appear in any of the columns
        assert all((filtered_entities == task.entities).apply(all))
        assert all(filtered_outcome == task.outcomes)

    def _test_load_task_without_exclusion(self, task_name, tasks_folder=None):
        if tasks_folder is None:
            tasks_folder = _get_test_tasks_folder()
        entities, outcomes = _load_task_definitions_from_folder(task_name, tasks_folder)

        excluded = []

        task = load_task_definition(
            task_name=task_name, exclude_symbols=excluded, tasks_folder=tasks_folder
        )

        # check that the excluded entities do not appear in any of the columns
        assert all((entities == task.entities).apply(all))
        assert all(outcomes == task.outcomes)

    def test_entities_task(self):
        task_name = "symbol_bin"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            tasks_folder=_get_test_tasks_folder(),
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            base_model=LogisticRegression(max_iter=2000, multi_class="auto"),
        )
        full_entity_task.run()
        assert not full_entity_task._cv_report is None

    def test_entities_task_overlap(self):
        task_name = "symbol_bin"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            tasks_folder=_get_test_tasks_folder(),
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            base_model=LogisticRegression(max_iter=2000, multi_class="auto"),
            overlap_entities=True,
        )
        full_entity_task.run()
        assert not full_entity_task._cv_report is None

        task_name = "simple_cat"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            scoring="roc_auc_ovo_weighted",
            base_model=LogisticRegression(
                max_iter=2000,
                multi_class="auto",
            ),
            tasks_folder=_get_test_tasks_folder(),
        )
        full_entity_task.run()
        assert not full_entity_task._cv_report is None

    def test_multiclass_task_midi(self):
        task_name = "simple_cat"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            base_model=LogisticRegression(max_iter=2000, multi_class="multinomial"),
            scoring="roc_auc_ovo_weighted",
            tasks_folder=_get_test_tasks_folder(),
        )
        full_entity_task.run()
        assert not full_entity_task._cv_report is None

    @pytest.mark.skip("slow test, run manually")
    def test_multiclass_task_extra_large(self):
        """When last tested, this took about 4.5 minutes to run."""
        task_name = "simple_cat"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            tasks_folder=_get_test_tasks_folder(),
            description_builder=NCBIDescriptor(),
            base_model=LogisticRegression(max_iter=2000, multi_class="multinomial"),
            scoring="roc_auc_ovo_weighted",
        )
        full_entity_task.run()
        minimal_succsess_rate = 0.5
        assert not full_entity_task._cv_report is None
        assert all(full_entity_task._cv_report["test_score"] >= minimal_succsess_rate)

    def test_task_modification_3D_to_2D(self):
        task_name = "interaction"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            tasks_folder=_get_test_tasks_folder(),
            description_builder=NCBIDescriptor(),
        )
        df_encode = PreComputedEncoder(_get_resources() / "file_embed_test.csv")
        encodes = df_encode.encode(
            pd.DataFrame([["C3orf18", "RPS2P45"], ["C3orf18", "RPS2P45"]])
        )
        assert full_entity_task._post_processing_mat(encodes).shape == (2, 5)

    def test_task_modification_2D_to_2D(self):
        task_name = "interaction"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        df_encode = PreComputedEncoder(_get_resources() / "file_embed_test.csv")
        encodes = df_encode.encode(pd.Series(["C3orf18", "RPS2P45"]))
        full_entity_task = EntitiesTask(
            task=task_name,
            tasks_folder=_get_test_tasks_folder(),
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
        )
        assert full_entity_task._post_processing_mat(encodes).shape == (2, 5)

    def test_convert_to_mat_series(self):
        df_encode = PreComputedEncoder(_get_resources() / "file_embed_test.csv")
        encodes = df_encode.encode(pd.Series(["C3orf18", "RPS2P45"]))
        coverted_encodings = convert_to_mat(encodes)
        assert isinstance(coverted_encodings, np.ndarray)
        assert coverted_encodings.shape == (2, len(encodes[0]))

    def test_convert_to_mat_df(self):
        df_encode = PreComputedEncoder(_get_resources() / "file_embed_test.csv")
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["C3orf18", "RPS2P45"], ["PLAC4", "PLAC4"]]),
                columns=["gene1", "gene2"],
            )
        )
        coverted_encodings = convert_to_mat(encodes)
        assert isinstance(coverted_encodings, np.ndarray)
        assert coverted_encodings.shape == (2, 2, 5)
        assert sum(coverted_encodings[0, 0, :] - encodes.iloc[0, 0]) == 0

    def test_convert_to_mat_df_1d(self):
        df_encode = PreComputedEncoder(_get_resources() / "file_embed_test.csv")
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array(["C3orf18", "RPS2P45"]),
                columns=["gene1"],
            )
        )
        coverted_encodings = convert_to_mat(encodes)
        assert isinstance(coverted_encodings, np.ndarray)
        assert coverted_encodings.shape == (2, len(encodes.iloc[0, 0]))

    def test_error_missing_task(self):
        with pytest.raises(ValueError, match="Couldn't find the task"):
            load_task_definition("no such test", tasks_folder=_get_test_tasks_folder())

    def test_load_task(self):
        task_def = load_task_definition(
            "symbol_bin", tasks_folder=_get_test_tasks_folder()
        )
        assert isinstance(task_def, TaskDefinition)

    def test_load_task_shape(self):
        task_def = load_task_definition(
            "symbol_bin", tasks_folder=_get_test_tasks_folder()
        )
        assert task_def.outcomes.shape == (100,)
        assert task_def.entities.shape == (100, 1)

    def test_entities_task_summary(self):
        task_name = "symbol_bin"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            tasks_folder=_get_test_tasks_folder(),
        )
        full_entity_task.run()
        summary = full_entity_task.summary()
        assert "encoder class" in summary
        assert "description class" in summary
        assert "mean_roc_auc" in summary
        assert "sample_size" in summary

    def test_list_subtests(self):
        task_name = "symbol_bin"
        tsk = load_task_definition(task_name, tasks_folder=_get_test_tasks_folder())
        entities, outcomes = tsk.entities, tsk.outcomes
        sub_entities, sub_outcomes = sub_sample_task_frames(
            entities, outcomes, frac=0.5
        )
        assert sub_entities.shape[0] == int(entities.shape[0] * 0.5)
        assert all(sub_entities.index == sub_outcomes.index)

    def test_get_task_names(self):
        tasks_folder = _get_tasks_folder()
        names = list(get_tasks_definition_names(tasks_folder))
        assert len(names) >= 65
        assert "RNA cancer distribution" in names
        assert "bivalent vs non-methylated" in names

    def test_task_modification_3D_to_2D_concat(self):
        task_name = "interaction"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            tasks_folder=_get_test_tasks_folder(),
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            encoding_post_processing="concat",
        )
        threeDmat = pd.DataFrame(
            data=[
                [[1, 2, 3, 4], [5, 6]],
                [[1, 2, 3, 4], [5, 6]],
                [[1, 2, 3, 4], [5, 6]],
            ]
        )
        assert full_entity_task._post_processing_mat(threeDmat).shape == (3, 6)

    def test_load_multilabel(self):
        task_name = "multi_label"
        task_definitions = load_task_definition(
            task_name,
            tasks_folder=_get_test_tasks_folder(),
        )
        assert isinstance(task_definitions.entities, pd.DataFrame)
        assert isinstance(task_definitions.outcomes, pd.DataFrame)
        assert task_definitions.entities.shape[0] == task_definitions.outcomes.shape[0]
        assert task_definitions.outcomes.shape[1] == 5

    def test_subsample(self):
        task_name = "symbol_bin"
        tsk = load_task_definition(task_name, tasks_folder=_get_test_tasks_folder())
        entities, outcomes = tsk.entities, tsk.outcomes
        sub_entities, sub_outcomes = sub_sample_task_frames(
            entities, outcomes, frac=0.3
        )
        assert sub_entities.shape[0] - entities.shape[0] * 0.3 < 1
        assert all(sub_entities.index == sub_outcomes.index)

    def test_subsample_multilable(self):
        task_name = "simple_cat"
        tsk = load_task_definition(task_name, tasks_folder=_get_test_tasks_folder())
        entities, outcomes = tsk.entities, tsk.outcomes
        sub_entities, sub_outcomes = sub_sample_task_frames(
            entities, outcomes, frac=0.27
        )
        assert sub_entities.shape[0] - (entities.shape[0] * 0.27) < 1
        assert all(sub_entities.index == sub_outcomes.index)

    def test_subsample_multilable_frac_1(self):
        task_name = "multi_label"
        tsk = load_task_definition(task_name, tasks_folder=_get_test_tasks_folder())
        entities, outcomes = tsk.entities, tsk.outcomes
        sub_entities, sub_outcomes = sub_sample_task_frames(
            entities, outcomes, frac=1.0
        )
        assert sub_entities.shape[0] == entities.shape[0]
        assert all(sub_entities.index == sub_outcomes.index)

    def test_sub_task_passed_though_entities_task(self):
        task_name = "multi_label"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"

        sub_task = "Outcomes_3"
        sub_outcome_task = EntitiesTask(
            task=task_name,
            tasks_folder=_get_test_tasks_folder(),
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            sub_task=sub_task,
        )

        assert isinstance(sub_outcome_task.task_definitions.entities, pd.DataFrame)
        assert isinstance(sub_outcome_task.task_definitions.outcomes, pd.Series)
        assert (
            sub_outcome_task.task_definitions.entities.shape[0]
            == sub_outcome_task.task_definitions.outcomes.shape[0]
        )

    def test_sub_task_wrong_name(self):
        task_name = "multi_label"

        sub_task = "no such sub_task"

        with pytest.raises(ValueError, match="Couldn't find the sub_task"):
            load_task_definition(
                task_name=task_name,
                tasks_folder=_get_test_tasks_folder(),
                sub_task=sub_task,
            )

    def test_sub_task_on_single_column_task(self):
        task_name = "symbol_bin"

        sub_task = "Outcomes"

        with pytest.raises(
            ValueError, match="only one outcome and can not be used with sub_tasks"
        ):
            load_task_definition(
                task_name=task_name,
                sub_task=sub_task,
                tasks_folder=_get_test_tasks_folder(),
            )

    def test_include(self):
        task_name = "symbol_bin"
        tsk = load_task_definition(task_name, tasks_folder=_get_test_tasks_folder())
        include = tsk.entities.sample(10).squeeze().values
        tsk_incl = load_task_definition(
            task_name, tasks_folder=_get_test_tasks_folder(), include_symbols=include
        )
        assert len(
            set(tsk_incl.entities.squeeze().values).intersection(set(include))
        ) == len(include)

    def test_include_interactions(self):
        task_name = "interaction"
        # we include two full lines and two lines with one of the two symbols.
        include = ["Gene_3", "Gene_80", "Gene_36", "Gene_15", "Gene_61", "Gene_91"]
        tsk_incl = load_task_definition(
            task_name, tasks_folder=_get_test_tasks_folder(), include_symbols=include
        )
        assert tsk_incl.entities.shape[0] == 2

    def test_entities_task_inclusion(self):
        task_name = "symbol_bin"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            tasks_folder=_get_test_tasks_folder(),
            include_symbols=["ATP6V0A1", "TUBG2", "MRPL43", "DHX8"],
        )
        assert full_entity_task.task_definitions.entities.shape[0] == 4

    def test_multiclass_task_with_th(self):
        task_name = "imbalanced_cat"
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        full_entity_task = EntitiesTask(
            task=task_name,
            encoder=mpnet_name,
            description_builder=NCBIDescriptor(),
            base_model=LogisticRegression(max_iter=5000),
            cv=5,
            scoring=["roc_auc_ovr_weighted"],
            tasks_folder=_get_test_tasks_folder(),
            cat_label_th=0.04,
        )
        full_entity_task.run()
        this_run_df = full_entity_task.summary()
        test_scores = this_run_df["test_roc_auc_ovr_weighted"].split(",")
        test_scores = list(map(float, test_scores))
        assert np.nan not in test_scores
