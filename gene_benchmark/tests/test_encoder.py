import unittest

import numpy as np
import pandas as pd
import pytest

from gene_benchmark.descriptor import (
    NCBIDescriptor,
)
from gene_benchmark.encoder import (
    MultiEntityEncoder,
    PreComputedEncoder,
    SentenceTransformerEncoder,
    create_random_embedding_matrix,
    randNone_dict,
)


class TestEncoder(unittest.TestCase):
    def test_description_and_encoder_allow_missing_and_random(self):
        descriptor = NCBIDescriptor()
        descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"])
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        encoded = encoder.encode(descriptions, randomize_missing=True)
        assert encoded.shape[0] == 4

    @pytest.mark.xfail(
        reason="This test needs to be fixed https://github.ibm.com/BiomedSciAI-Innersource/gene-benchmark/issues/6"
    )
    def test_random_encodings_missing_genes_descriptions(self):
        descriptor = NCBIDescriptor(allow_partial=False)
        descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"])
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        unique_entities = encoder._get_unique_entities(descriptions)
        index_of_missing = (
            [unique_entities.index(None)] if None in unique_entities else []
        )
        unique_encodings_before_random = encoder.encoder.encode(unique_entities)
        unique_encodings_with_random = encoder._get_encoding(
            unique_entities, randomize_missing=True
        )
        for missing_ind in index_of_missing:
            assert not np.array_equal(
                unique_encodings_before_random[missing_ind],
                unique_encodings_with_random[missing_ind],
            )

    def test_get_unique_entities_series(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        unique_entities = df_encode._get_unique_entities(
            pd.Series(["PLAC4", "RPS2P45", "PLAC4"])
        )
        assert len(unique_entities) == 2

    def test_get_unique_entities_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        unique_entities = df_encode._get_unique_entities(
            pd.DataFrame(
                np.array([["PLAC4", "RPS2P45"], ["PLAC4", "C3orf18"]]),
                columns=["gene1", "gene2"],
            )
        )
        assert len(unique_entities) == 3

    def test_PreComputedEncoder_series(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(pd.Series(["PLAC4", "RPS2P45", "C3orf18"]))
        assert encodes.shape[0] == 3

    def test_PreComputedEncoder_dataframeload(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv",
            encoder_model=pd.read_csv(
                "gene_benchmark/tests/resources/file_embed_test.csv",
                index_col="symbol",
            ),
        )
        encodes = df_encode.encode(pd.Series(["PLAC4", "RPS2P45", "C3orf18"]))
        assert encodes.shape[0] == 3

    def test_PreComputedEncoder_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["PLAC4", "RPS2P45"], ["PLAC4", "C3orf18"]]),
                columns=["gene1", "gene2"],
            )
        )
        assert encodes.shape == (2, 2)

    def test_PreComputedEncoder_allocation_series(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(pd.Series(["PLAC4", "RPS2P45", "C3orf18"]))
        plac4_values = df_encode.model.loc["PLAC4"].values
        assert sum(plac4_values - encodes[0]) == 0

    def test_PreComputedEncoder_allocation_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["PLAC4", "RPS2P45"], ["PLAC4", "C3orf18"]]),
                columns=["gene1", "gene2"],
            )
        )
        plac4_values = df_encode.model.loc["PLAC4"].values
        assert sum(plac4_values - encodes.iloc[0][0]) == 0
        assert sum(encodes.iloc[0][0] - encodes.iloc[1][0]) == 0

    def test_PreComputedEncoder_allocation_mixed_order_series(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(pd.Series(["RPS2P45", "PLAC4", "C3orf18"]))
        plac4_values = df_encode.model.loc["PLAC4"].values
        assert sum(plac4_values - encodes[1]) == 0

    def test_PreComputedEncoder_allocation_mixed_order_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["C3orf18", "RPS2P45"], ["PLAC4", "PLAC4"]]),
                columns=["gene1", "gene2"],
            )
        )
        plac4_values = df_encode.model.loc["PLAC4"].values
        assert sum(plac4_values - encodes.iloc[1][0]) == 0
        assert sum(encodes.iloc[1][1] - encodes.iloc[1][0]) == 0

    def test_PreComputedEncoder_missing_genes_false(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        with pytest.raises(ValueError, match="Couldn't encode the entities"):
            df_encode.encode(
                pd.Series(["PLAC4", "RPS2P45", "C3orf18", "IAMNOTAGENE"]),
                allow_missing=False,
            )

    def test_PreComputedEncoder_missing_genes_false_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        with pytest.raises(ValueError, match="Couldn't encode the entities"):
            df_encode.encode(
                pd.DataFrame(
                    np.array([["C3orf18", "RPS2P45"], ["PLAC4", "IAMNOTAGENE"]]),
                    columns=["gene1", "gene2"],
                ),
                allow_missing=False,
            )

    def test_PreComputedEncoder_None_good_shape(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.Series(["PLAC4", "RPS2P45", "C3orf18", None]),
            allow_missing=True,
            randomize_missing=True,
        )
        assert encodes[3].shape[0] == df_encode.model.shape[1]

    def test_PreComputedEncoder_missing_genes_allow(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.Series(["PLAC4", "RPS2P45", "C3orf18", "IAMNOTAGENE", None]),
            allow_missing=True,
            randomize_missing=True,
        )
        assert not np.all(np.isnan(encodes.iloc[-1]))
        assert encodes.shape[0] == 5

    def test_PreComputedEncoder_missing_genes_allow_with_nan(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.Series(["PLAC4", "RPS2P45", "C3orf18", "IAMNOTAGENE"]),
            allow_missing=True,
            randomize_missing=False,
        )
        assert np.all(np.isnan(encodes.iloc[-1]))

    def test_PreComputedEncoder_missing_genes_allow_df(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["C3orf18", "RPS2P45"], ["PLAC4", "IAMNOTAGENE"]]),
                columns=["gene1", "gene2"],
            ),
            allow_missing=True,
            randomize_missing=False,
        )
        assert np.all(np.isnan(encodes.iloc[-1][-1]))

    def test_PreComputedEncoder_missing_genes_allow_df_with_randomize(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        encodes = df_encode.encode(
            pd.DataFrame(
                np.array([["C3orf18", "RPS2P45"], ["PLAC4", "IAMNOTAGENE"]]),
                columns=["gene1", "gene2"],
            ),
            allow_missing=True,
            randomize_missing=True,
        )
        assert not np.all(np.isnan(encodes.iloc[-1][-1]))

    def test_sentenceTransformerEncoder_series(self):
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        txt1 = "It was originally taken from a Latin text written by a Roman Scholar, Sceptic and Philosopher\
            by the name of Marcus Tullius Cicero, who influenced the Latin language greatly."
        txt2 = (
            "The filler text we know today has been altered over the years (in fact Lorem isn't actually a\
            Latin word. It is suggested that the reason that the text starts with Lorem is because there was a\
                page break spanning the word"
        )
        txt3 = "Do-lorem. If you a re looking for a translation of the text, it's meaningless. The original text\
            talks about the pain and love involved in the pursuit of pleasure or something like that."
        txt4 = "The reason we use Lorem Ipsum is simple. If we used real text, it would possibly distract from\
            the DESIGN of a page (or indeed, might even be mistakenly inappropriate. Or if we used something like \
                Insert Text Here..., this would also distract from the design. Using Lorem Ipsum allows us to SEE \
                    the design without being distracted by readable or unrealistic text."
        encoded = encoder.encode(pd.Series([txt1, txt2, txt3, txt4]))
        assert encoded.shape[0] == 4

    def test_sentenceTransformerEncoder_series_with_none(self):
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        txt1 = "It was originally taken from a Latin text written by a Roman Scholar, Sceptic and Philosopher\
            by the name of Marcus Tullius Cicero, who influenced the Latin language greatly."
        txt2 = (
            "The filler text we know today has been altered over the years (in fact Lorem isn't actually a\
            Latin word. It is suggested that the reason that the text starts with Lorem is because there was a\
                page break spanning the word"
        )
        txt3 = "Do-lorem. If you a re looking for a translation of the text, it's meaningless. The original text\
            talks about the pain and love involved in the pursuit of pleasure or something like that."
        txt4 = "The reason we use Lorem Ipsum is simple. If we used real text, it would possibly distract from\
            the DESIGN of a page (or indeed, might even be mistakenly inappropriate. Or if we used something like \
                Insert Text Here..., this would also distract from the design. Using Lorem Ipsum allows us to SEE \
                    the design without being distracted by readable or unrealistic text."
        txt5 = None
        encoded = encoder.encode(
            pd.Series([txt1, txt2, txt3, txt4, txt5]), randomize_missing=True
        )
        assert encoded.shape[0] == 5

    def test_entenceTransformerEncoder_df(self):
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        txt1 = "It was originally taken from a Latin text written by a Roman Scholar, Sceptic and Philosopher\
            by the name of Marcus Tullius Cicero, who influenced the Latin language greatly."
        txt2 = (
            "The filler text we know today has been altered over the years (in fact Lorem isn't actually a\
            Latin word. It is suggested that the reason that the text starts with Lorem is because there was a\
                page break spanning the word"
        )
        txt3 = "Do-lorem. If you a re looking for a translation of the text, it's meaningless. The original text\
            talks about the pain and love involved in the pursuit of pleasure or something like that."
        txt4 = "The reason we use Lorem Ipsum is simple. If we used real text, it would possibly distract from\
            the DESIGN of a page (or indeed, might even be mistakenly inappropriate. Or if we used something like \
                Insert Text Here..., this would also distract from the design. Using Lorem Ipsum allows us to SEE \
                    the design without being distracted by readable or unrealistic text."
        encoded = encoder.encode(
            pd.DataFrame(
                np.array([[txt1, txt2], [txt3, txt4]]), columns=["gene1", "gene2"]
            )
        )
        assert encoded.shape == (2, 2)

    def test_sentenceTransformerEncoder_pdseries(self):
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        txt1 = "It was originally taken from a Latin text written by a Roman Scholar, Sceptic and Philosopher\
            by the name of Marcus Tullius Cicero, who influenced the Latin language greatly."
        txt2 = (
            "The filler text we know today has been altered over the years (in fact Lorem isn't actually a\
            Latin word. It is suggested that the reason that the text starts with Lorem is because there was a\
                page break spanning the word"
        )
        txt3 = "Do-lorem. If you a re looking for a translation of the text, it's meaningless. The original text\
            talks about the pain and love involved in the pursuit of pleasure or something like that."
        txt4 = "The reason we use Lorem Ipsum is simple. If we used real text, it would possibly distract from\
            the DESIGN of a page (or indeed, might even be mistakenly inappropriate. Or if we used something like \
                Insert Text Here..., this would also distract from the design. Using Lorem Ipsum allows us to SEE \
                    the design without being distracted by readable or unrealistic text."
        encoded = encoder.encode(
            pd.Series(data=[txt1, txt2, txt3, txt4], index=["1", "2", "3", "4"])
        )
        assert encoded.shape[0] == 4

    def test_create_random_embedding_matrix(self):
        mat = create_random_embedding_matrix(rows=2, cols=10)
        assert mat.shape == (2, 10)

    def test_descriptions_and_encoder_partial_and_random(self):
        descriptor = NCBIDescriptor(allow_partial=False)
        descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"])
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        encoded = encoder.encode(descriptions, randomize_missing=True)
        assert encoded.shape[0] == 4

    def test_none_dict(self):
        rn_dict = randNone_dict(
            np.random.normal, {"size": (1, 100)}, {"hello": "world"}
        )
        assert rn_dict["hello"] == "world"
        r1 = rn_dict[None]
        r2 = rn_dict[None]
        assert r1.shape[1] == 100
        assert (r1 != r2).all()

    def test_descriptions_and_encoder_partial_and_random_diff(self):
        descriptor = NCBIDescriptor(allow_partial=False)
        descriptions = descriptor.describe(
            pd.Series(["BRCA1", "FOXP2", "NOTAGENENAME", "PLAC4"])
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        encoded = encoder.encode(descriptions, randomize_missing=True)
        assert (encoded[3] != encoded[2]).all()

    def test_PreComputedEncoder_Summary(self):
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        df_encode.encode(pd.Series(["PLAC4", "RPS2P45", "PLAC4", "IAMNOTAGENE"]))
        expected_summary = {
            "encoder_model_name": "gene_benchmark/tests/resources/file_embed_test.csv",
            "encoder class": "PreComputedEncoder",
            "num_of_missing_entities": 1,
            "allow_missing": True,
            "randomize_missing": True,
        }
        assert expected_summary == df_encode.summary()

    def test_sentencetransormer_summary(self):
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        encoder.encode(
            pd.Series(["BRCA1", "FOXP2", None, "PLAC4"]), randomize_missing=True
        )
        expected_summary = {
            "encoder class": "SentenceTransformerEncoder",
            "encoder_model_name": mpnet_name,
        }
        assert encoder.summary() == expected_summary

    def test_multi_entity(self):
        to_encode = pd.DataFrame(
            np.array([["PLAC4", "PLAC4"], ["RPS2P45", "RPS2P45"]]),
            columns=["symbol_mpnet", "symbol_df"],
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        df_encode = PreComputedEncoder(
            "gene_benchmark/tests/resources/file_embed_test.csv"
        )
        enc_dict = {"symbol_mpnet": encoder, "symbol_df": df_encode}
        ml_enc = MultiEntityEncoder(enc_dict)
        ml_encode = ml_enc.encode(to_encode)

        assert ml_encode["symbol_df"][0].shape[0] == 5
        assert ml_encode["symbol_mpnet"][1].shape[0] == 768
        df_code = df_encode.encode(pd.Series(["PLAC4", "RPS2P45"]))
        assert all(df_code[0] == ml_encode["symbol_df"][0])

    def test_multi_entity_as_single(self):
        to_encode = pd.DataFrame(
            np.array([["PLAC4", "PLAC4"], ["RPS2P45", "RPS2P45"]]),
            columns=["symbol_mpnet", "symbol_mpnet"],
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        PreComputedEncoder("gene_benchmark/tests/resources/file_embed_test.csv")
        enc_dict = {"symbol_mpnet": encoder}
        ml_enc = MultiEntityEncoder(enc_dict)
        ml_encode = ml_enc.encode(to_encode)
        assert ml_encode["symbol_mpnet"].values[0, 0].shape[0] == 768
        assert ml_encode["symbol_mpnet"].shape == (2, 2)

    def test_multi_entity_mis_col_error(self):
        to_encode = pd.DataFrame(
            np.array([["PLAC4", "PLAC4"], ["RPS2P45", "RPS2P45"]]),
            columns=["symbol_mpnet", "symbol_gggg"],
        )
        mpnet_name = "sentence-transformers/all-mpnet-base-v2"
        encoder = SentenceTransformerEncoder(mpnet_name)
        PreComputedEncoder("gene_benchmark/tests/resources/file_embed_test.csv")
        enc_dict = {"symbol_mpnet": encoder}
        ml_enc = MultiEntityEncoder(enc_dict)
        with pytest.raises(Exception, match="columns which are not in the encoding"):
            ml_enc.encode(to_encode)
