# wrangling
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# local packages
from ..preprocessing import process_dataset
from ..frameworks import sklearn
from ..evaluation import evaluate_model

__all__ = [
    "run_experiment_bow_svm",
    "run_experiment_ada_svm",
]


def run_experiment_bow_svm(dataset: DatasetDict, test_dataset: Dataset = None) -> dict:
    x_train, y_train, x_test, y_test = process_dataset(dataset, features="text")

    # overwrite the test set if using ood dataset
    if test_dataset is not None:
        x_test, y_test = test_dataset["text_clean"], test_dataset["labels"]

    pipe = Pipeline(
        [
            ("vectoriser", sklearn.get_vectoriser()),
            ("clf", MultiOutputClassifier(sklearn.get_sgd_classifier())),
        ]
    )
    params = {
        "vectoriser__max_features": [5_000, 10_000, 30_000],
        "clf__estimator__alpha": np.logspace(-5, 0, 6),
    }
    grid = sklearn.run_grid_search(pipe, params, x_train, y_train)
    pipe = grid.best_estimator_
    evaluate_model(pipe, x_train, y_train, name="train")
    results = evaluate_model(pipe, x_test, y_test, name="test")
    return results


def run_experiment_ada_svm(
    dataset: DatasetDict,
    test_dataset: Dataset = None,
) -> dict:
    x_train, y_train, x_test, y_test = process_dataset(dataset, features="embeddings")

    # overwrite the test set if using ood dataset
    if test_dataset is not None:
        x_test, y_test = test_dataset["embeddings"], test_dataset["labels"]

    clf = MultiOutputClassifier(sklearn.get_sgd_classifier())
    params = {
        "estimator__alpha": np.logspace(-5, 0, 6),
    }
    grid = sklearn.run_grid_search(clf, params, x_train, y_train)
    pipe = grid.best_estimator_
    evaluate_model(pipe, x_train, y_train, name="train")
    results = evaluate_model(pipe, x_test, y_test, name="test")
    return results
