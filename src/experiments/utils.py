# standard library
from typing import Callable

# wrangling
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight

# monitoring
import mlflow


__all__ = [
    "count_parameters",
    "get_class_weight",
    "log_experiment",
]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_class_weight(y: list[list[int]]) -> list[float]:
    class_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(range(1, 18)),
        y=[label for labels in y for label in labels],
    )
    return class_weight


def log_experiment(
    dataset: DatasetDict,
    experiment_func: Callable,
    run_name: str,
    test_dataset: Dataset = None,
    **kwargs
):
    metrics = experiment_func(dataset=dataset, test_dataset=test_dataset)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("size-train", dataset["train"].num_rows)
        mlflow.log_param("size-test", dataset["test"].num_rows)
        for k, v in kwargs.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
