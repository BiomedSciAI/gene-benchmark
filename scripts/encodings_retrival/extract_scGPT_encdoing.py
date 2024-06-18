import json
from pathlib import Path

import click
import gdown
import numpy as np
import pandas as pd
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

MODEL_URLS = {
    "args.json": "https://drive.google.com/file/d/1y4UJVflGl-b2qm-fvpxIoQ3XcC2umjj0/view?usp=drive_link",
    "vocab.json": "https://drive.google.com/file/d/127FdcUyY1EM7rQfAS0YI4ms6LwjmnT9J/view?usp=drive_link",
    "best_model.pt": "https://drive.google.com/file/d/1MJaavaG0ZZkC_yPO4giGRnuCe3F1zt30/view?usp=drive_link",
}


def download_google_drive_file(url, file_name, output_dir):
    file_id = url.split("/d/")[1].split("/view")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    output = output_dir + "/" + file_name
    gdown.download(download_url, output, quiet=False)


def get_vocabulary(vocab_file: str) -> GeneVocab:
    """
    read the vocabulary file.

    Args:
    ----
        vocab_file (str): path to file

    Returns:
    -------
        GeneVocab: the vocabulary

    """
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    return vocab


def create_empty_model(model_config_file: str, vocab: GeneVocab) -> TransformerModel:
    """
    Create an empty model matching the config file and the loaded vocabulary
    model parameters needed for model initialization are mostly taken from the saved config file.

    Args:
    ----
        model_config_file (str): path to model config file
        vocab (GeneVocab): the vocabulary

    Returns:
    -------
        TransformerModel: an empty model that can be filled from the saved weights

    """
    # Retrieve model parameters from config files
    with open(model_config_file) as f:
        model_configs = json.load(f)

    model = TransformerModel(
        len(vocab),  # size of vocabulary
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        vocab=vocab,
        pad_value=-2,
        n_input_bins=51,
    )
    return model


def load_scgpt_embedding(model_dir):
    """
    Retrieve the data-independent gene embeddings from scGPT:
    This method does the main part of the job.  The input is a path to a directory containing the
    scGPT model.
    This specific model saving structure matches that of the scGPT pretrained "blood model"
    taken from https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU
    The model itself comes in three files:
        * vocab.json:  the vocabulary with links the symbols (gene names) and the token IDs
        * args.json:  Parameters to override the default values for model shapes and such
        * best_model.json:  The actual model.
    The method has three parts:
        1) Extracting the mapping from token id to symbol
        2) loading the model from files
        3) extracting the encoding for each token id and packing it with the symbols in a DataFrame.

    Args:
    ----
        model_dir (str|Path): path to the directory containing the model

    Returns:
    -------
        pd.DataFrame: A dataframe that maps symboles to an embedding (as a vector of length 512)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(model_dir)

    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    # read vocabulary file
    vocab = get_vocabulary(vocab_file)
    # map from gene name to gene idx.
    gene2idx = vocab.get_stoi()

    # upload the model from the files:
    model = create_empty_model(model_config_file=model_config_file, vocab=vocab)
    # load the model via torch
    torch_model = torch.load(model_file, map_location=torch.device("cpu"))
    # upload the parameters from the torch model
    model.load_state_dict(torch_model, strict=False)

    # extract the list of gene and matching token ids
    gene_symbols = list(gene2idx.keys())
    gene_ids = np.array([gene2idx[k] for k in gene_symbols])
    # extract the encoding vectors for the ids above from the model
    gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))

    # move the embedding into a data frame so we can use merge to add the symbols to the vectors
    gene_embeddings_df = pd.DataFrame(
        gene_embeddings.detach().cpu().numpy(), index=gene_symbols
    )
    return gene_embeddings_df


@click.command()
@click.option(
    "--allow-downloads",
    "-l",
    type=click.BOOL,
    help=f"download files directly from {MODEL_URLS.values()}",
    default=False,
)
@click.option(
    "--input-file-dir",
    type=click.STRING,
    help="The path to the directory with the data files, if no files exists and the user will download from a url, this is the path to the directory the files will be saved to",
    default="./encodings/scGPT",
)
@click.option(
    "--output-file-name",
    "-m",
    type=click.STRING,
    help="the root directory the encodings will be saved to.",
    default="./encodings/scGPT/encodings.csv",
)
def main(allow_downloads, input_file_dir, output_file_name):
    if allow_downloads:
        for file_name, url in MODEL_URLS.items():
            download_google_drive_file(url, file_name, output_dir=input_file_dir)

    embedding = load_scgpt_embedding(model_dir=input_file_dir)
    embedding.to_csv(output_file_name)


if __name__ == "__main__":
    main()
