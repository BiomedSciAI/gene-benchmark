from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from .descriptor import add_prefix_to_dict


def _break_string(string: str, max_length: int) -> list[str]:
    """
    given a string returns a list where each element is at most max_length
        where the last element can be shorter.

    Args:
    ----
        string (str): the string to break into list
        max_length (int): the maximal size of each piece

    Returns:
    -------
        list[str]: list with the string where each element is at most max_length and the last is shorter

    """
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]


class randNone_dict(dict):
    """An dict extension class that returns method return value each time that the key None is used."""

    def __init__(self, none_method, method_args, user_dict):
        """
        Inits the class with a regular dictionary and the None return method.

        Args:
        ----
            rand_method : A method to be activated each time the None key is used
            user_dict (dict): The dictionary to be used as the base dictionary

        """
        self.none_method = none_method
        self.method_args = method_args
        super().__init__(user_dict)

    def __getitem__(self, key):
        """
        If the None key is used the return value will be the return value of the None method.

        Args:
        ----
            key : the dictionary key

        Returns:
        -------
            value : the value stored for the key

        """
        if key is None:
            return self.none_method(**self.method_args)
        else:
            return super().__getitem__(key)


def create_random_embedding_matrix(rows, cols):
    """
    Create random embedding matrix from normal distribution.

    Args:
    ----
    rows: number or rows
    cols: number of cols

    Returns: np array with random embeddings

    """
    return np.random.normal(size=(rows, cols))


class Encoder(ABC):
    """An interface for a  entity encoder  class."""

    @abstractmethod
    def encode(
        self, entities, allow_missing=True, randomize_missing=True, random_len=None
    ) -> pd.Series:
        """
        Encode unique gene descriptions.

        Args:
        ----
            entities : a pd.Series | pd.DataFrame containing gene symbols.
            allow_missing : a flag for handling gene symbols with no existing embeddings.
            relevant to the PreComputedEncoder class.
            randomize_missing :  (bool) : If False missing encodings will result in NANs. If True random embeddings will be used.

        Returns:
        -------
            a pd.Series | pd.DataFrame containing description embeddings for the provided gene symbols.

        """
        pass

    def summary(self) -> dict:
        """
        Returns dictionary with summary details on the encoder.

        Returns
        -------
            dict: dictionary with summary details on the encoder

        """
        return {
            "encoder class": self.__class__.__name__,
        }


class MultiEntityEncoder(Encoder):
    """
    Multi Entity encoders enable encoding using multiple encoders.

    Args:
    ----
        Encoder (_type_): _description_

    """

    def __init__(self, encoder_dict={}) -> None:
        """
        Initiates an encoder capable of combining multiple encoders.

        Args:
        ----
            encoder_dict (dict, optional): Each key contain the name
            of the encoding type and the value is the encoder this
            correspond to the encoded data which will need to have
            corresponding column names. Defaults to {}.

        """
        self.encoder_dict = encoder_dict

    def encode(
        self, entities, allow_missing=True, randomize_missing=True, random_len=None
    ):
        self._raise_unknown_encoding_columns(entities)
        encodings = pd.DataFrame(index=entities.index, columns=entities.columns)
        for encoding_type in self.encoder_dict:
            encodings[encoding_type] = self.encoder_dict[encoding_type].encode(
                entities[encoding_type],
                allow_missing=allow_missing,
                randomize_missing=randomize_missing,
                random_len=random_len,
            )
        return encodings

    def _raise_unknown_encoding_columns(self, entities):
        missing_cols = set(entities.columns) - set(self.encoder_dict.keys())
        if len(missing_cols) > 0:
            raise Exception(
                "The encoding DataFrame has columns which are not in the encoding dictionary."
                f" Missing cols are: {','.join(missing_cols)}"
            )
        return

    def summary(self):
        """
        Returns dictionary with summary details on the encoder.

        Returns
        -------
            dict: dictionary with summary details on the encoder

        """
        all_entities_summary_dict = super().summary()
        for encoding_type in self.encoder_dict:
            summary_dict = self.encoder_dict[encoding_type].summary()
            summary_dict_with_prefix = add_prefix_to_dict(
                encoding_type, summary_dict, underscore_separator=True
            )
            all_entities_summary_dict.update(summary_dict_with_prefix)

        return all_entities_summary_dict


class SingleEncoder(Encoder):
    """An interface for a single entity encoder  class."""

    def __init__(self, embedding_model_name) -> None:
        self.embedding_model_name = embedding_model_name

    def encode(
        self, entities, allow_missing=True, randomize_missing=True, random_len=None
    ):
        #  A None may cause an issue in sentence_transformers/models/Transformer.py#L121
        unique_entities = list(filter(None, self._get_unique_entities(entities)))
        if len(unique_entities) > 0:
            unique_encodings = self._get_encoding(
                unique_entities,
                allow_missing=allow_missing,
                randomize_missing=randomize_missing,
            )
        else:
            unique_encodings = [None]
        encoding_dict = self._get_encoding_dict(
            unique_entities,
            unique_encodings,
            use_rand_dict=randomize_missing,
            random_len=random_len,
        )
        mapped_encodings = self._allocate_encoding(entities, encoding_dict)
        return mapped_encodings

    def _get_unique_entities(self, entities):
        """
        Get unique entities names to prevent duplicate NCBI description extraction.

        Args:
        ----
            entities : a series/df with the gene symbol names

        returns: list with unique gene symbol names

        """
        return list(set(entities.values.flatten()))

    @abstractmethod
    def _get_encoding(self, entities, allow_missing):
        pass

    def summary(self):
        """
        Returns dictionary with summary details on the encoder.

        Returns
        -------
            dict: dictionary with summary details on the encoder

        """
        return {
            "encoder class": self.__class__.__name__,
            "encoder_model_name": self.encoder_model_name,
        }

    def _get_encoding_dict(
        self, unique_entities, unique_encodings, use_rand_dict=False, random_len=None
    ):
        """
        Create gene entities:encoding dictionary for reallocating to input series/dataframe.

        Args:
        ----
            unique_entities : a the unique gene symbol names
            unique_encodings : the encodings for the corresponding gene names

        returns: dictionary with gene symbol names and encodings that returns a fresh random value for each None key retrieval

        """
        rnd_len = len(unique_encodings[0]) if random_len is None else random_len
        base_dict = dict(zip(unique_entities, unique_encodings))
        base_dict[None] = None
        if use_rand_dict:
            rand_embedding_length = rnd_len
            return randNone_dict(
                np.random.normal, {"size": rand_embedding_length}, base_dict
            )
        else:
            return base_dict

    def _allocate_encoding(self, entities, encoding_dict):
        """
        Map the entities encodings to the corresponding entity, note that we are using 'map' and not 'replace'
        because the allocation is 1:encoding dimension. the 'replace' works for allocation of 1:1.

        Args:
        ----
            entities: the original imputed series/df with gene names
            encoding_dict : the encoding dictionary created in _get_encoding_dict

        returns: the original entities data structure with the embeddings of the gene symbols

        """
        return entities.map(lambda x: encoding_dict[x])

    def _raise_unable_to_encode(self, entities):
        raise ValueError(f"Couldn't encode the entities: {','.join(entities)} ")


class PreComputedEncoder(SingleEncoder):
    """
    Load pre existing encodings for given entities from csv file.

    Args:
    ----
        Encoder (_type_): _description_

    """

    def __init__(
        self,
        encoder_model_name: str,
        encoder_model: pd.DataFrame = None,
        read_csv_params: dict = None,
    ):
        if read_csv_params is None:
            read_csv_params = {"index_col": "symbol"}

        if encoder_model is None:
            self.model = pd.read_csv(encoder_model_name, **read_csv_params)
        else:
            self.model = encoder_model
        self.encoder_model_name = encoder_model_name
        self.missing_entities = None

    def _get_encoding(self, entities, allow_missing=True, randomize_missing=False):
        """
        Check if requested entities exist and return corresponding entities encodings
        in the same order as in the entities object. note - the re-indexing of the encodings
        by the imputed entities object is important for the creation if the allocation dictionary.

        Args:
        ----
            entities : a list with the gene symbol names
            allow_missing (bool) : If False an error will be raised for pre computed encodings
            randomize_missing (bool) : If False missing encodings will result in NANs. If True random embeddings will be used.

        returns: pre computed gene symbol encodings in the same order as the entities

        """
        existing_entities = list(filter(lambda x: x in self.model.index, entities))
        missing_entities = list(filter(lambda x: x not in self.model.index, entities))
        self.missing_entities = missing_entities
        self.allow_missing = allow_missing
        self.randomize_missing = randomize_missing
        self._validate_encoding(missing_entities, allow_missing)
        encodings = self.model.loc[existing_entities].reindex(entities).values
        if randomize_missing:
            index_of_missing = np.where(np.isin(entities, missing_entities))[0]
            encodings[index_of_missing] = create_random_embedding_matrix(
                rows=len(index_of_missing), cols=encodings.shape[1]
            )
        return encodings

    def _validate_encoding(self, missing_entities, allow_missing):
        """
        Check if missing items are permitted.

        Args:
        ----
            missing_entities : a list of entities names that are not in the csv file
            allow_missing : bool provided by the user if missing entities are permitted

        Raises: error if unable to encode

        """
        if len(missing_entities) > 0 and not allow_missing:
            self._raise_unable_to_encode(missing_entities)

    def summary(self, short=True):
        """
        Generate summary.

        Args:
        ----
            short (bool, optional): Short summary does not include the missing entities. Defaults to True.

        Returns:
        -------
            dict: summary dictionary

        """
        summary_dict = {}
        summary_dict.update(super().summary())
        if self.missing_entities:
            if not short:
                summary_dict["missing_entities"] = ",".join(
                    [str(v) for v in self.missing_entities]
                )
            summary_dict["num_of_missing_entities"] = len(self.missing_entities)
            summary_dict["allow_missing"] = self.allow_missing
            summary_dict["randomize_missing"] = self.randomize_missing
        return summary_dict


class SentenceTransformerEncoder(SingleEncoder):
    """encode a list of descriptions into numeric vectors using sentence transformers."""

    def __init__(
        self,
        encoder_model_name: str = None,
        encoder_model: SentenceTransformer = None,
        show_progress_bar: bool = True,
        batch_size: int = 32,
    ):
        if encoder_model is None:
            self.encoder = SentenceTransformer(encoder_model_name)
        else:
            self.encoder = encoder_model
        self.encoder_model_name = encoder_model_name
        self.num_of_missing = None
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

    def _get_encoding(self, entities, **kwargs):
        """
        Encode descriptions. For missing descriptions the generated embedding can be randomized.

        Args:
        ----
            entities : a list of entities
            allow_missing (bool) : bool provided by the user if missing entities are permitted
            randomize_missing (bool) : If True for missing descriptions randomized embeddings will be generated.

        returns: the encodings matrix for the entities

        The entities should never contain a None, as a bug in sentence_transformers/models/Transformer.py#L121
        can be triggered when the first text of a batch is None (error raised from line 133)

        """
        assert (
            None not in entities
        ), "A downstream bug will crash on encoding None sometimes, so there should never be a None here."

        encodings = self.encoder.encode(
            entities,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
        )
        return encodings

    def summary(self):
        summary_dict = super().summary()
        if self.num_of_missing:
            summary_dict["num_of_missing"] = self.num_of_missing
        return summary_dict


class BERTEncoder(SingleEncoder):
    """encode a list of descriptions into numeric vectors using transformers BERT encoders."""

    def __init__(
        self,
        encoder_model_name: str = None,
        tokenizer_name: str = None,
        trust_remote_code: bool = False,
        context_size: int = None,
    ):
        config = BertConfig.from_pretrained(encoder_model_name)
        self.encoder = AutoModel.from_pretrained(
            encoder_model_name, trust_remote_code=trust_remote_code, config=config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        self.encoder_model_name = encoder_model_name
        self.tokenizer_name = tokenizer_name
        self.trust_remote_code = trust_remote_code
        self.context_size = context_size
        super().__init__(encoder_model_name)

    def _get_encoding(self, entities, **kwargs):
        return list(map(self._encode_multiple_contexts, entities))

    def _encode_multiple_contexts(self, ent):
        return np.mean(list(map(self._encode_single_entry, _break_string(ent))), axis=0)

    def _encode_single_entry(self, ent):
        inputs = self.tokenizer(ent, return_tensors="pt")["input_ids"]
        hidden_states = self.encoder(inputs)[0]
        return torch.mean(hidden_states[0], dim=0).detach()

    def summary(self):
        summary_dict = super().summary()
        if self.num_of_missing:
            summary_dict["num_of_missing"] = self.num_of_missing
        summary_dict["tokenizer_name"] = self.tokenizer_name
        return summary_dict
