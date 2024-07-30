import unittest

import numpy as np
import pandas as pd

from gene_benchmark.descriptor import (
    CSVDescriptions,
    MultiEntityTypeDescriptor,
    NaiveDescriptor,
    NCBIDescriptor,
    _gene_symbol_to_ensemble_ids,
    add_prefix_to_dict,
    missing_col_or_nan,
)

disease_csv_path_for_testing = "gene_benchmark/tests/resources/csv2prompt.csv"


class TestDescriptor(unittest.TestCase):
    def test_getting_gene_num_descriptions(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(pd.Series(["BRCA1", "FOXP2"]))
        assert len(descriptions) == 2

    def test_getting_gene_num_descriptions_with_summary(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(pd.Series(["BRCA1", "FOXP2"]))
        assert all("summary" in x for x in descriptions)

    def test_getting_gene_num_descriptions_without_summary(self):
        descriptor = NCBIDescriptor(add_summary=False)
        descriptions = descriptor.describe(pd.Series(["BRCA1", "FOXP2"]))
        assert all("summary" not in x for x in descriptions)

    def test_getting_gene_num_descriptions_get_unique(self):
        descriptor = NCBIDescriptor()
        unique = descriptor._get_unique_entities(pd.Series(["BRCA1", "FOXP2", "BRCA1"]))
        assert len(unique) == 2

    def test_getting_gene_num_descriptions_multiple_series(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(pd.Series(["BRCA1", "FOXP2", "BRCA1"]))
        assert len(descriptions) == 3

    def test_getting_gene_num_descriptions_multiple_frame(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol"], index=[1, 2, 3], data=["BRCA1", "FOXP2", "BRCA1"]
            )
        )
        assert len(descriptions) == 3
        assert descriptions.iloc[0].values == descriptions.iloc[-1].values

    def test_getting_gene_num_descriptions_multiple_dim_dataframe(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(
            pd.DataFrame(
                np.array([["PLAC4", "RPS2P45"], ["PLAC4", "C3orf18"]]),
                columns=["gene1", "gene2"],
            )
        )
        assert descriptions.shape == (2, 2)
        assert descriptions.iloc[0, 0] == descriptions.iloc[1, 0]

    def test_raise_missing_gene(self):
        descriptor = NCBIDescriptor()
        with self.assertWarns(UserWarning):
            descriptor.describe(
                pd.Series(["BRCA1", "FOXP2", "BRCA1", "NOTAGENENAME"]),
                allow_missing=False,
            )

    def test_not_raise_missing_gene(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME"])
        )
        assert len(descriptions) == 3

    @staticmethod
    def get_disease_description(csv_path=disease_csv_path_for_testing):
        """
            unility function for quickly getting a desease descriptions maker
        Args:
            csv_path (str|Path, optional): path to csv file. Defaults to None.
        """
        return CSVDescriptions(csv_file_path=csv_path, index_col="id")

    def test_disease_description(self):
        descriptor = self.get_disease_description(csv_path=disease_csv_path_for_testing)
        descriptions = descriptor.describe(
            pd.Series(["EFO_0004254", "EFO_0005853", "Orphanet_122"])
        )
        assert len(descriptions) == 3

    def test_disease_description_bad_name(self):
        descriptor = self.get_disease_description()
        with self.assertWarns(UserWarning):
            descriptor.describe(
                pd.Series(
                    ["EFO_0004254", "EFO_0005853", "Orphanet_122", "NOTADISEASE"]
                ),
                allow_missing=False,
            )

    def test_disease_description_finds_missing(self):
        descriptor = self.get_disease_description()
        entities = pd.Series(
            ["EFO_0004254", "EFO_0005853", "Orphanet_122", "NOTADISEASE"]
        )
        descriptions = descriptor.describe(
            entities,
            allow_missing=True,
        )
        bad = descriptor.get_missing_entities(entities, None)
        assert len(bad) == 1
        assert len(descriptions) == len(entities)

    def test_is_partial_row(self):
        descriptor = NCBIDescriptor()
        descriptions_df = descriptor._retrieve_dataframe_for_entities(
            ["PLAC4", "IAMNOTAGENE", "C3orf18"]
        )

        res = [
            descriptor.is_partial_description_row(df_row)
            for ind, df_row in descriptions_df.iterrows()
        ]
        assert res[0]
        assert res[1]
        assert not res[2]

    def test_test_partial_description(self):
        descriptor = NCBIDescriptor()
        descriptions_df = descriptor._retrieve_dataframe_for_entities(
            ["PLAC4", "IAMNOTAGENE", "C3orf18"]
        )
        res = [
            descriptor._manage_row_generation(df_row)
            for ind, df_row in descriptions_df.iterrows()
        ]
        assert res[0] is None
        assert res[1] is None
        assert not res[2] is None

    def test_test_partial_get_description(self):
        descriptor = NCBIDescriptor(allow_partial=False)
        descriptions_df = descriptor.describe(
            pd.Series(["PLAC4", "IAMNOTAGENE", "C3orf18"])
        )
        assert descriptions_df[0] is None
        assert descriptions_df[1] is None
        assert not descriptions_df[2] is None

    def test_test_partial_get_partial_description(self):
        descriptor = NCBIDescriptor(allow_partial=True)
        descriptions_df = descriptor.describe(
            pd.Series(["PLAC4", "IAMNOTAGENE", "C3orf18"]), allow_missing=True
        )
        assert not descriptions_df[0] is None
        assert descriptions_df[1] is None
        assert not descriptions_df[2] is None

    def test_test_partial_get_partial_no_missing(self):
        descriptor = NCBIDescriptor(allow_partial=True)
        descriptions_df = descriptor.describe(
            pd.Series(["PLAC4", "IAMNOTAGENE", "C3orf18"]), allow_missing=False
        )
        assert not descriptions_df[0] is None
        assert not descriptions_df[2] is None
        assert len(descriptions_df.index) == 2

    def test_test_partial_get_partial_no_missing_df(self):
        descriptor = NCBIDescriptor(allow_partial=True)
        descriptions_df = descriptor.describe(
            pd.DataFrame(
                [["PLAC4", "C3orf18", "C3orf18"], ["PLAC4", "IAMNOTAGENE", "C3orf18"]]
            ),
            allow_missing=False,
        )
        assert all(descriptions_df.notna().sum())
        assert len(descriptions_df.index) == 1

    def test_missing_col_or_nan(self):
        missing_series = pd.Series(
            index=["test_header", "another_header"], data=[np.nan, "value"]
        )
        assert not missing_col_or_nan(missing_series, "another_header")
        assert missing_col_or_nan(missing_series, "test_header")
        assert missing_col_or_nan(missing_series, "missing_header")

    def test_description_maker_summary(self):
        descriptor = NCBIDescriptor(allow_partial=False)
        descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"]), allow_missing=True
        )
        summary = descriptor.summary()
        expected_summary = {
            "allow_partial": False,
            "num_missing_entities": 1,
            "allow_missing": True,
            "description class": "NCBIDescriptor",
            "description columns": "summary,name,symbol",
        }
        assert summary == expected_summary

    def test_csvdescription_maker_summary(self):
        descriptor = self.get_disease_description(csv_path=disease_csv_path_for_testing)
        descriptor.describe(
            pd.Series(["EFO_0004254", "EFO_0005853", "Orphanet_122", "NOTADISEASE"])
        )
        summary = descriptor.summary()
        expected_summary = {
            "allow_partial": True,
            "num_missing_entities": 1,
            "allow_missing": True,
            "description class": "CSVDescriptions",
            "csv_file_path": "gene_benchmark/tests/resources/csv2prompt.csv",
        }
        assert summary == expected_summary

    def test_csvdescription_return_string(self):
        descriptor = self.get_disease_description(csv_path=disease_csv_path_for_testing)
        descriptions = descriptor.describe(
            pd.Series(["EFO_0004254", "EFO_0005853", "Orphanet_122"])
        )
        assert isinstance(descriptions[2], str)

    def test_MultiEntityTypeDescriptor_descriptions(self):
        csv_path = "gene_benchmark/tests/resources/disease_descriptions.csv"
        description_dict = {
            "symbol": NCBIDescriptor(allow_partial=False),
            "disease": CSVDescriptions(csv_file_path=csv_path, index_col="id"),
        }
        descriptor = MultiEntityTypeDescriptor(description_dict=description_dict)
        descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol", "disease"],
                data=[("BRCA1", "cancer"), ("PLAC4", "als")],
            )
        )
        assert descriptions.shape == (2, 2)
        assert descriptions.iloc[1, 1] == "this is a even more bad disease"

    def test_MultiEntityTypeDescriptor_descriptions_one_entity_type(self):
        description_dict = {
            "symbol": NCBIDescriptor(allow_partial=False),
        }
        descriptor = MultiEntityTypeDescriptor(description_dict=description_dict)
        descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol"],
                data=[("BRCA1"), ("PLAC4")],
            )
        )
        assert descriptions.shape == (2, 1)
        assert descriptions.iloc[1, 0] == None

    def test_MultiEntityTypeDescriptor_descriptions_multiple_columns(self):
        csv_path = "gene_benchmark/tests/resources/disease_descriptions.csv"
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
        assert descriptions.shape == (2, 3)
        assert descriptions.iloc[1, 1] == "this is a even more bad disease"
        assert descriptions.iloc[1, 2] == None
        assert descriptions.iloc[0, 1] == "cancer is a very bad disease"

    def test_MultiEntityTypeDescriptor_descriptions_multiple_description_types(self):
        csv_path = "gene_benchmark/tests/resources/disease_descriptions.csv"
        description_dict = {
            "symbol": NCBIDescriptor(allow_partial=False),
            "disease": CSVDescriptions(csv_file_path=csv_path, index_col="id"),
            "disease2": CSVDescriptions(csv_file_path=csv_path, index_col="id"),
        }
        descriptor = MultiEntityTypeDescriptor(description_dict=description_dict)
        descriptions = descriptor.describe(
            pd.DataFrame(
                columns=["symbol", "disease", "disease2"],
                data=[("BRCA1", "cancer", "crones"), ("PLAC4", "als", "crones")],
            )
        )
        assert descriptions.shape == (2, 3)
        assert descriptions.iloc[1, 1] == "this is a even more bad disease"
        assert descriptions.iloc[1, 0] == None
        assert descriptions.iloc[0, 1] == "cancer is a very bad disease"

    def test_MultiEntityTypeDescriptor_summary(self):
        csv_path = "gene_benchmark/tests/resources/disease_descriptions.csv"
        description_dict = {
            "symbol": NCBIDescriptor(allow_partial=False),
            "disease": CSVDescriptions(csv_file_path=csv_path, index_col="id"),
        }
        descriptor = MultiEntityTypeDescriptor(description_dict=description_dict)
        descriptor.describe(
            pd.DataFrame(
                columns=["symbol", "disease"],
                data=[("BRCA1", "cancer"), ("PLAC4", "als")],
            )
        )
        descriptions_summary = descriptor.summary()
        assert descriptions_summary["symbol_allow_partial"] == False
        assert descriptions_summary["disease_csv_file_path"] == csv_path

    def test_add_prefix_to_dict(self):
        d = {"a": 1, "b": 2}
        new_d = add_prefix_to_dict("z", d, underscore_separator=True)
        assert list(new_d.keys()) == ["z_a", "z_b"]

    def test_test_NaiveDescriptor(self):
        descriptor = NaiveDescriptor()
        entities = pd.Series(data=["PLAC4", "IAMNOTAGENE", "C3orf18"])
        des = descriptor.describe(entities)
        assert all(des == entities)

    def test_symbol_to_ensemble():
        gene_symbols = ["BRCA1", "TP53", "EGFR", "NOTGENE"]
        ensembles = _gene_symbol_to_ensemble_ids(gene_symbols)
        real_vals = {
            "BRCA1": "ENSG00000012048",
            "TP53": "ENSG00000141510",
            "EGFR": "ENSG00000146648",
            "NOTGENE": None,
        }
        assert real_vals == ensembles
