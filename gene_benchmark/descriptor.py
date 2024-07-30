import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import mygene
import pandas as pd
import requests


def _gene_symbol_to_ensemble_ids(
    symbols: list[str], species: str = "human"
) -> dict[str, str]:
    """
    converts between symbols to their ensemble ids.

    Args:
    ----
        symbols (list[str]): A list of gene symbols
        species (str, optional): The species for the ensemble id conversion. Defaults to "human".

    Returns:
    -------
        dict[str, str]: dictionary with symbol ensemble pairs (if symbol didn't find a ensemble id it will not appear)

    """
    mg = mygene.MyGeneInfo()
    query = " ".join(symbols)
    gene_info = mg.query(query, fields="symbol,ensembl.gene", species=species)
    ids = {}
    for hit in gene_info["hits"]:
        symbol = hit["symbol"]
        ensembl_id = hit.get("ensembl", {}).get("gene", None)
        ids[symbol] = ensembl_id
    return ids


def _fetch_ensembl_sequence(ensembl_gene_id):
    """
    retries the base pair sequence of a given ensemble id using ensembl.org API.

    Args:
    ----
        ensembl_gene_id (str): an ensemble id

    Returns:
    -------
        str: base pair sequence of the gene

    """
    if not ensembl_gene_id:
        return None
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_gene_id}?content-type=text/plain"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


def missing_col_or_nan(df_series, indx):
    """
    Checks if df_series is missing index or has it with a null values.

    Args:
    ----
        df_series : The series to examine
        indx : the index name

    Returns:
    -------
        bool: true if df_series doesn't have a index entry or if it is a null value

    """
    if indx in df_series.index:
        return pd.isna(df_series[indx])
    else:
        return True


def add_prefix_to_dict(prefix: str, dictionary: dict, underscore_separator: bool):
    """
    Returns the dictionary but when every key has the prefix prefix.

    Args:
    ----
        prefix (str): the prefix string
        dictionary (dict): a dictionary
        underscore_separator (bool): if true an underscore will be added between the prefix and the key

    Returns:
    -------
        dictionary (dict): a dictionary with the keys updated with the prefix

    """
    if underscore_separator:
        prefix = f"{prefix}_"
    return {prefix + key: value for key, value in dictionary.items()}


class TextDescriptor(ABC):
    """An interface for description retrieving classes."""

    @abstractmethod
    def summary(self):
        """
        Return the class summary
        Returns:
            dict: dictionary with descriptors.
        """

    @abstractmethod
    def describe(
        self,
        entities: pd.DataFrame | pd.Series,
        allow_missing=True,
        first_description_only=False,
    ):
        """A method for extracting the entities descriptions."""


class NaiveDescriptor(TextDescriptor):
    """
    A naive descriptor which keeps the input id (symbols etc)
    as they are without modifying them, the purpose of the class is to
    combine pre computed encodings that operate on symbols or id's
    and text based encoding such as diseases.


    """

    def summary(self):
        return {
            "description class": self.__class__.__name__,
        }

    def describe(
        self,
        entities: pd.DataFrame | pd.Series,
        allow_missing=True,
        first_description_only=False,
    ):
        """
        _summary_.

        Args:
        ----
            entities (pd.DataFrame | pd.Series): the entities to describes
            allow_missing (bool, optional): For compatibility reasons
            first_description_only (bool, optional): For compatibility reasons

        Returns:
        -------
            pd.DataFrame | pd.Series: returns the entities as is

        """
        return entities


class SingleEntityTypeDescriptor(TextDescriptor):
    def __init__(self, allow_partial: bool = False) -> None:
        self.missing_entities = None
        self.allow_missing = None
        self.allow_partial = allow_partial

    @abstractmethod
    def _retrieve_dataframe_for_entities(
        self, entities: list, first_description_only=False
    ):
        """
        Extract information from the KB on the entities, to be used for description building.

        Args:
        ----
            entities (iterable | pd.DataFrame | pd.Series): gene symbols for extraction
            first_description_only (bool): If false can return multiple descriptions for genes otherwise will return the first description

        Returns:
        -------
            A pd.DataFrame with the entity names as index and the knowledge in the rest of the columns.
            Returns a row for each entry in the entities list.

        """

    def summary(self, short=True):
        """
        Return the class summary.

        Args:
        ----
            short (bool): short does not include the missing entities themselves
        Returns:
            dict: dictionary with descriptors.

        """
        sum_dict = {
            "allow_partial": self.allow_partial,
            "allow_missing": self.allow_missing,
            "description class": self.__class__.__name__,
        }
        if not short:
            sum_dict["missing_entities"] = self.missing_entities
        sum_dict["num_missing_entities"] = (
            len(self.missing_entities) if self.missing_entities else 0
        )
        return sum_dict

    @abstractmethod
    def get_missing_entities(self, elements, element_metadata_df):
        """
        Get a list of elements in the entities list for which no information was found in the knowledge base.
        If the element_metadata_df is None, will use the data frame read from the CSV in the init.

        Args:
        ----
        elements: list of elements to check if missing
        element_metadata_df (DataFrame): data frame with knowledge

        Returns:
        -------
        list of elements for which no data was found

        """

    @abstractmethod
    def row_to_description(self, df_row: pd.Series) -> str:
        """
        Converts a entries DataFrame row into a description.

        Args:
        ----
            df_row : a data frame row with name symbol and summary columns

        Returns:
        -------
            str: the description

        """

    @abstractmethod
    def is_partial_description_row(self, df_row: pd.Series) -> bool:
        """
        Method to check if row has all the values we need for full description generation.

        Args:
        ----
            row (pd.Series): the row to check

        Returns:
        -------
            bool

        """

    def _warn_missing_entities(self, entities):
        """
        Deal with cases where some entries are missing in the data.  Used when allow_missing is off.

        Args:
        ----
            entities (iterable[str]): the missing entries

        Raises:
        ------
            ValueError

        """
        warnings.warn(
            f"Couldn't create description for the entities: {','.join(entities)} ",
            UserWarning,
        )

    def _manage_row_generation(self, df_row: pd.Series):
        """
        Manage partial rows:  if allow partial is false and the row is partial, return none.
        otherwise use the row_to_description to generate the description.

        Args:
        ----
            df_row : a data frame row with name symbol and summary columns x

        Returns:
        -------
            Returns None if the description can not be created.
            return the description if it can

        """
        if not self.allow_partial and self.is_partial_description_row(df_row):
            return None
        else:
            return self.row_to_description(df_row)

    def describe(
        self,
        entities: pd.DataFrame | pd.Series,
        allow_missing=True,
        first_description_only=False,
    ):
        """
        Given a pandas series or DataFrame containing entities (genes, disease, etc) symbols,
        extracts the unique symbols. Then generate or extract the description and place them back to the series or DataFrame.
        if multiple descriptions are available in the data the first will be used (when first_description_only is true).

        Args:
        ----
            entities (pd.DataFrame | pd.Series): The set of symbols to generate description for
            allow_missing (bool): If False a warning will be raised with the missing symbol names and the returning df will not contain them,
                                    otherwise it will return None
            description (bool): If false can return multiple descriptions for genes otherwise will return the first description

        Returns:
        -------
            pd.DataFrame | pd.Series: A pandas DataFrame or series with description per gene symbol

        """
        unique_entities = self._get_unique_entities(entities)
        gene_metadata_df = self._retrieve_dataframe_for_entities(
            pd.Series(unique_entities), first_description_only=first_description_only
        )
        missing = self.get_missing_entities(entities, gene_metadata_df)
        self.missing_entities = missing
        self.allow_missing = allow_missing
        descriptions = self._construct_descriptions(gene_metadata_df)
        unique_descriptions_dict = descriptions.to_dict()
        entities_descriptions = entities.replace(unique_descriptions_dict)
        if len(missing) > 0 and not allow_missing:
            self._warn_missing_entities(missing)
            entities_descriptions = entities_descriptions.dropna()
        return entities_descriptions

    def _construct_descriptions(self, gene_metadata_df):
        """
        Give a DataFrame of knowledge, contract descriptions for each line using the row_to_description function.

        Args:
        ----
            element_metadata_df (DataFrame): data frame with knowledge

        Returns:
        -------
            pd.Series: the names as keys and the descriptions as values.

        """
        return gene_metadata_df.apply(self._manage_row_generation, axis=1)

    def _get_unique_entities(self, entities: pd.DataFrame | pd.Series | Iterable):
        """
        Get unique entities names to prevent duplicate KB description extractions.

        Args:
        ----
            entities (pd.DataFrame | pd.Series): list of entries with possible duplicated

        Raises:
        ------
            TypeError: if input is wrong type

        Returns:
        -------
            list[str]: a list of the entries after removing the duplicates.

        """
        if isinstance(entities, pd.DataFrame | pd.Series):
            return list(set(entities.values.flatten()))
        if isinstance(entities, Iterable):
            return list(entities)
        raise TypeError("entities should be of type pandas DataFrame or pandas Series.")


class NCBIDescriptor(SingleEntityTypeDescriptor):
    """
    Creates descriptions for gene symbol based on data from the NCBI.
    the descriptions are build as follows:
    "Gene symbol {symbol} full name {name_txt} with the summary {summary_txt}.
    """

    def __init__(self, allow_partial=False, add_summary=True) -> None:
        """
        Initialize descriptor class.

        Args:
        ----
            allow_partial (bool, optional):if true a partial description can be returned if false it will return None if the row is
            missing name, symbol or summary. Defaults to False.
            is_partial_row_function (callable): function to identify if a row only has partial knowledge.

        """
        super().__init__(allow_partial=allow_partial)
        if add_summary:
            self.needed_columns = ["summary", "name", "symbol"]
        else:
            self.needed_columns = ["name", "symbol"]
        self.add_summary = add_summary
        self.is_partial_row_function = self.is_partial_description_row

    def _retrieve_dataframe_for_entities(
        self, entities: list, first_description_only=False
    ):
        """
        Extract information from NCBI on the entities, to be used for description building.
        This implementation accesses the data online.

        Args:
        ----
        entities (iterable | pd.DataFrame | pd.Series): gene symbols for extraction
        first_description_only (bool): If false can return multiple descriptions for genes otherwise will return the first description

        Returns:
        -------
        A pd.DataFrame with the entity names as index and the knowledge in the rest of the columns.
        Returns a row for each entry in the entities list.

        """
        mg = mygene.MyGeneInfo()
        gene_metadata_df = mg.querymany(
            entities,
            scopes="symbol",
            species="human",
            fields=",".join(self.needed_columns),
            as_dataframe=True,
            verbose=False,
        )
        if not first_description_only:
            gene_metadata_df = gene_metadata_df[
                ~gene_metadata_df.index.duplicated(keep="first")
            ]
        return gene_metadata_df

    def row_to_description(self, df_row: pd.Series):
        """
        Converts a NCBI DataFrame row into a description.

        Args:
        ----
            df_row : a data frame row with name symbol and summary columns x

        Returns:
        -------
            str : " Gene symbol - {'symbol'} full name {name_txt} with the summary {summary_txt}"
            Returns None if the description can not be created.

        """
        name_txt = (
            df_row["name"] if not pd.isna(df_row["name"]) else "No available name"
        )
        if self.add_summary:
            summary_txt = (
                df_row["summary"]
                if not pd.isna(df_row["summary"])
                else "No available summary"
            )
        if all(pd.isna(df_row[v]) for v in self.needed_columns):
            return None
        else:
            if self.add_summary:
                description = f"Gene symbol {df_row['symbol']} full name {name_txt} with the summary {summary_txt}"
            else:
                description = f"Gene symbol {df_row['symbol']} full name {name_txt}"
            return description

    def get_missing_entities(self, entities, element_metadata_df):
        """
        Get a list of elements in the entities list for which no information was found in the knowledge base.

        Args:
        ----
        entities: list of entities to check if missing
        element_metadata_df (DataFrame): data frame with knowledge

        Returns:
        -------
        list of elements for which no data was found

        """
        if "notfound" in element_metadata_df.columns:
            missing_symbols = element_metadata_df.loc[
                element_metadata_df["notfound"] == True, :
            ].index.values
            return list(missing_symbols)
        return []

    def is_partial_description_row(self, row: pd.Series):
        """
        Default function to check if row has all the values we need for full description generation.

        Args:
        ----
            row (pd.Series): the row to check

        Returns:
        -------
            bool

        """
        return has_missing_columns(row, column_names=self.needed_columns)

    def summary(self):
        summary_dict = super().summary()
        summary_dict.update(
            {
                "description columns": ",".join(self.needed_columns),
            }
        )
        return summary_dict


class CSVDescriptions(SingleEntityTypeDescriptor):
    """
    Creates descriptions for gene symbol based on data from the a CSV file.
    the descriptions are build by the description_generation_function.
    """

    def __init__(
        self,
        csv_file_path,
        index_col="id",
        description_generation_function: callable = None,
        allow_partial: bool = True,
        is_partial_row_function: callable = None,
    ):
        """
        Read the data frame from the file, and set the csv-raw to description function.

        Args:
        ----
            csv_file_path (str|Path): path to csv file with the knowledge
            index_col (str, optional): name of index column to use from the csv. Defaults to "id".
            description_generation_function (callable, optional): function to be called on each raw to generate the descriptions.
                 Defaults to the build in `default_row_to_description` function which takes the precomputed descriptions from
                 the 'descriptions' column

        """
        super().__init__(allow_partial=allow_partial)
        if description_generation_function:
            self.row_to_description_function = description_generation_function
        else:
            self.row_to_description_function = self.default_row_to_description

        if is_partial_row_function:
            self.is_partial_row_function = is_partial_row_function
        else:
            self.is_partial_row_function = self._default_is_partial_row_function
        self.csv_file_path = csv_file_path
        self.index_col = index_col
        self.source_data_frame = pd.read_csv(csv_file_path, index_col=index_col)

    def _retrieve_dataframe_for_entities(
        self, entities: list, first_description_only=False
    ):
        """
        Extract the entity summary for the imputed entities from a csv.

        Args:
        ----
            entities (iterable | pd.DataFrame | pd.Series): gene symbols for extraction
            first_description_only (bool): If false can return multiple descriptions for genes otherwise will return the first description

        Returns:
        -------
            An array with the descriptions for the corresponding gene entities.

        """
        overlap = list(filter(lambda x: x in self.source_data_frame.index, entities))
        result_df = pd.DataFrame(index=overlap, columns=self.source_data_frame.columns)
        result_df = self.source_data_frame.loc[overlap, :]
        if not first_description_only:
            result_df = result_df[~result_df.index.duplicated(keep="first")]
        return result_df

    def row_to_description(self, df_row: pd.Series) -> str:
        return self.row_to_description_function(df_row)

    def default_row_to_description(self, df_row: pd.Series):
        """
        Converts a entries dataframe row into a description.

        Args:
        ----
            df_row : a data frame row with name symbol and summary columns

        Returns:
        -------
            str : description Gene symbol - {'symbol'} full name {name_txt} with the summary {summary_txt}"

        """
        return df_row["description"]

    def get_missing_entities(self, entities, element_metadata_df):
        """
        Get a list of elements in the entities list for which no information was found in the knowledge base.
        If the element_metadata_df is None, will use the data frame read from the CSV in the init.

        Args:
        ----
            elements: list of elements to check if missing
            element_metadata_df (DataFrame): data frame with knowledge

        Returns:
        -------
            list of elements for which no data was found

        """
        if element_metadata_df is None:
            element_metadata_df = self.source_data_frame
        return list(set(entities) - set(element_metadata_df.index.array))

    def is_partial_description_row(self, df_row: pd.Series) -> bool:
        """
        _summary_.

        Args:
        ----
            df_row (df.Series): _description_

        Returns:
        -------
            bool: _description_

        """
        return self.is_partial_row_function(df_row)

    def _default_is_partial_row_function(self, _):
        return False

    def summary(self):
        summary_dict = super().summary()
        summary_dict.update(
            {
                "csv_file_path": self.csv_file_path,
            }
        )
        return summary_dict


class MultiEntityTypeDescriptor(TextDescriptor):
    def __init__(
        self, description_dict: dict[str, SingleEntityTypeDescriptor] | None = None
    ) -> None:
        self.description_dict = description_dict
        if self.description_dict is None:
            self.description_dict = {}

    def describe(
        self,
        entities: pd.DataFrame | pd.Series,
        allow_missing=True,
        first_description_only=False,
    ):
        for entity_type in self.description_dict:
            entities[entity_type] = self.description_dict[entity_type].describe(
                entities[entity_type],
                allow_missing=allow_missing,
                first_description_only=first_description_only,
            )
        return entities

    def summary(self):
        """Returns the dictionary with the summaries of all entities."""
        all_entities_summary_dict = {}
        for entity_type in self.description_dict:
            summary_dict = self.description_dict[entity_type].summary()
            summary_dict_with_prefix = add_prefix_to_dict(
                entity_type, summary_dict, underscore_separator=True
            )
            all_entities_summary_dict.update(summary_dict_with_prefix)

        return all_entities_summary_dict


def construct_open_targets_disease_descriptions(x):
    """
    Function that can be plugged in as a row-to-description function to generate a description
       currently now used, mostly here as an example.

    Args:
    ----
        x (pd.Series): input row

    Returns:
    -------
        str: the description

    """
    return f"{x['id']}:0 full name {x['name']} described as: {x['description']}"


def has_missing_columns(df_row, column_names):
    """
    Checks if the knowledge df row has the components to create a full description or not.
    The list of columns should match the row-to-description function used.

    Args:
    ----
        df_row (pd.Series): a data frame row
        column_names (list): A list of all of the columns required to create a description

    Returns:
    -------
        bool: if the partial_list columns exists with a non  null value

    """
    return any(missing_col_or_nan(df_row, col) for col in column_names)


class BasePairDescriptor(SingleEntityTypeDescriptor):

    def __init__(self, specie: str = "human", description_col="description") -> None:
        self.species = specie
        self.description_col = description_col
        self.missing_entities = None
        self.allow_missing = None
        self.allow_partial = True

    def _retrieve_dataframe_for_entities(
        self, entities: list, first_description_only=False
    ):
        ensembles = _gene_symbol_to_ensemble_ids(entities, species=self.species)
        base_pairs = {
            symbol: _fetch_ensembl_sequence(ensemble)
            for symbol, ensemble in ensembles.items()
        }
        pair_bse_df = pd.DataFrame(index=entities, columns=[self.description_col])
        pair_bse_df[self.description_col] = [
            base_pairs[v] if v in base_pairs else None for v in pair_bse_df.index
        ]
        return pair_bse_df

    def get_missing_entities(self, elements, element_metadata_df):
        return list(
            filter(lambda x: x not in element_metadata_df.dropna().index, elements)
        )

    def row_to_description(self, df_row: pd.Series) -> str:
        return df_row[self.description_col]

    def is_partial_description_row(self, df_row: pd.Series) -> bool:
        return False
