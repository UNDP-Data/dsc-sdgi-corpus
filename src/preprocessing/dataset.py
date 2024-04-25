# standard library
from typing import Literal

# wrangling
import numpy as np
from datasets import DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = [
    "process_dataset",
]


def process_dataset(
    dataset: DatasetDict, features: Literal["text", "embeddings"]
) -> tuple:
    if features == "text":
        x_train = np.asarray(dataset["train"]["text_clean"])
        x_test = np.asarray(dataset["test"]["text_clean"])
    elif features == "embeddings":
        x_train = np.vstack(dataset["train"]["embedding"])
        x_test = np.vstack(dataset["test"]["embedding"])
    else:
        raise ValueError(f"Unknown features {features}.")
    binariser = MultiLabelBinarizer(classes=range(1, 18))
    y_train = binariser.fit_transform(dataset["train"]["labels"])
    y_test = binariser.transform(dataset["test"]["labels"])
    return x_train, y_train, x_test, y_test
