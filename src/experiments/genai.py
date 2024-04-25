# wrangling
import numpy as np
from datasets import DatasetDict

# utils
from tqdm import tqdm

# local packages
from ..frameworks import classify_text_zeroshot


__all__ = [
    "run_experiment_zeroshot_gpt",
]


def run_experiment_zeroshot_gpt(dataset: DatasetDict) -> dict:
    predictions = []
    for text in tqdm(dataset["test"]["text"]):
        try:
            labels = classify_text_zeroshot(text)
        except Exception as e:
            print(e)
            labels = np.zeros(17)
        predictions.append(labels)
    df = dataset["test"].to_pandas().reindex(["labels", "metadata"], axis=1)
    df["prediction"] = predictions
    df["language"] = df["metadata"].str.get("language")
    df["size"] = df["metadata"].str.get("size")
    df.drop("metadata", axis=1, inplace=True)
    df.to_json("gpt-predictions.jsonl", orient="records", lines=True)
    return {"predictions": True}
