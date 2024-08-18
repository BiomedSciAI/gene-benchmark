from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from gene_benchmark.descriptor import (
    BasePairDescriptor,
    CSVDescriptions,
    MultiEntityTypeDescriptor,
    NaiveDescriptor,
    NCBIDescriptor,
)
from gene_benchmark.encoder import (
    BERTEncoder,
    MultiEntityEncoder,
    PreComputedEncoder,
    SentenceTransformerEncoder,
)


def load_class(class_name, class_args=None):
    """
    a factory given a class name and it's arguments returns an initiated class.

    Args:
    ----
        class_name (str): the class name to initiate
        class_args (dictionary, optional): class_name arguments dictionary. Defaults to None.

    Returns:
    -------
        : a class_name initiated with class_args

    """
    if class_args:
        return type_dict[class_name](**class_args)
    else:
        return type_dict[class_name]()


def get_gene_disease_description(
    csv_file_path, csv_type="diseaseId", naive_descriptor=False
):
    if naive_descriptor:
        return MultiEntityTypeDescriptor(
            {
                csv_type: CSVDescriptions(csv_file_path=csv_file_path, index_col="id"),
                "symbol": NaiveDescriptor(),
            }
        )
    else:
        return MultiEntityTypeDescriptor(
            {
                csv_type: CSVDescriptions(csv_file_path=csv_file_path, index_col="id"),
                "symbol": NCBIDescriptor(),
            }
        )


def get_gene_disease_multi_encoder(
    csv_file_path,
    encoder_model_name,
    sentence_id="diseaseId",
    csv_type="symbol",
):
    return MultiEntityEncoder(
        {
            csv_type: PreComputedEncoder(encoder_model_name=csv_file_path),
            sentence_id: SentenceTransformerEncoder(encoder_model_name),
        }
    )


type_dict = {
    "SentenceTransformerEncoder": SentenceTransformerEncoder,
    "NCBIDescriptor": NCBIDescriptor,
    "CSVDescriptions": CSVDescriptions,
    "PreComputedEncoder": PreComputedEncoder,
    "GeneDiseaseDescriptions": get_gene_disease_description,
    "LinearRegression": LinearRegression,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "get_gene_disease_multi_encoder": get_gene_disease_multi_encoder,
    "BasePairDescriptor": BasePairDescriptor,
    "Multilayer_Perceptron_classifier": MLPClassifier,
    "Multilayer_Perceptron_regressor": MLPRegressor,
    "BERTEncoder": BERTEncoder,
}
