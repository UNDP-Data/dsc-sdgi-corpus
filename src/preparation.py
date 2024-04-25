# wrangling
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

# utils
from tqdm import tqdm

# local packages
from .preprocessing import preprocess_example
from .frameworks import get_embeddings

__all__ = [
    "prepare_dataset",
    "prepare_ood_dataset",
]


def prepare_dataset(size: str, language: str) -> DatasetDict:
    sizes = ["s", "m", "l"] if size == "x" else [size]
    languages = ["en", "es", "fr"] if language == "xx" else [language]
    dataset = load_dataset("UNDP/sdgi-corpus")
    dataset = dataset.filter(
        lambda x: x["metadata"]["size"] in sizes
        and x["metadata"]["language"] in languages
    )
    dataset = dataset.map(preprocess_example)
    return dataset


def prepare_ood_dataset() -> Dataset:
    """
    Prepare dataset for a dataset from SDG-Meter for out-of-domain (OOD) testing.
    Source: https://github.com/UNEP-Economy-Division/SDG-Meter.

    Returns
    -------
    dataset : Dataset
        Concatenation of train and val splits of the OOD dataset.
    """
    data = {}
    for split in ("train", "val"):
        url = f"https://raw.githubusercontent.com/UNEP-Economy-Division/SDG-Meter/main/data/{split}_sample.csv"
        df = pd.read_csv(url, index_col=0)
        df["labels"] = df.loc[:, "SDG1":"SDG17"].values.tolist()
        df.rename({"ID": "id"}, axis=1, inplace=True)
        df = df.reindex(["id", "text", "labels"], axis=1)
        data[split] = Dataset.from_pandas(df, preserve_index=False)
    dataset = DatasetDict(**data)

    dataset = concatenate_datasets([dataset['train'], dataset['val']])
    df = dataset.to_pandas()
    embeddings = []

    for text in tqdm(df['text']):
        embedding = get_embeddings([text])[0]
        embeddings.append(embedding)
    df['embeddings'] = embeddings

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.map(preprocess_example)
    return dataset
