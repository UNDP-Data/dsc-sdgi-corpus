# standard library
from collections import Counter

# wrangling
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader

# local packages
from ..preprocessing import process_dataset
from ..frameworks import MultilabelTextDataset, SAGEGraph, train_model
from ..evaluation import evaluate_gnn
from ..utils import GOALS
from . import utils

__all__ = [
    "run_experiment_gnn",
]


def run_experiment_gnn(dataset: DatasetDict, test_dataset: Dataset = None) -> dict:
    input_dim = 128
    hidden_dim = 256
    epochs = 30
    learning_rate = 0.002
    weight_decay = 5e-5
    vocabulary_size = 30_000

    # fix the seed for reproducibility
    torch.manual_seed(0)
    counter = Counter()
    for text in dataset["train"]["text_clean"]:
        counter.update(text.split())
    vocabulary = sorted(token for token, _ in counter.most_common(vocabulary_size))
    class_weights = utils.get_class_weight(dataset["train"]["labels"])
    x_train, y_train, x_test, y_test = process_dataset(dataset, features="text")

    # overwrite the test set if using ood dataset
    if test_dataset is not None:
        x_test, y_test = test_dataset["text_clean"], test_dataset["labels"]

    dataset_train = MultilabelTextDataset(x_train, y_train, vocabulary=vocabulary)
    dataset_test = MultilabelTextDataset(x_test, y_test, vocabulary=vocabulary)
    dataset_train, dataset_valid, _ = split_dataset(
        dataset=dataset_train, frac_list=[0.8, 0.2, 0.0], shuffle=True, random_state=42
    )
    dataloader_train = GraphDataLoader(
        dataset=dataset_train,
        batch_size=16,
        drop_last=False,
        shuffle=True,
    )
    dataloader_valid = GraphDataLoader(
        dataset=dataset_valid,
        batch_size=16,
        drop_last=False,
        shuffle=True,
    )
    dataloader_test = GraphDataLoader(
        dataset=dataset_test,
        batch_size=32,
        drop_last=False,
        shuffle=False,
    )
    model = SAGEGraph(
        vocab_size=len(vocabulary),
        in_feats=input_dim,
        h_feats=hidden_dim,
        num_classes=len(GOALS),
    )
    print(f"Parameter count: {utils.count_parameters(model):,}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model = train_model(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        train_dataloader=dataloader_train,
        valid_dataloader=dataloader_valid,
        class_weights=np.asarray(class_weights),
        patience=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    evaluate_gnn(model, dataloader_train, name="train")
    results = evaluate_gnn(model, dataloader_test, name="test")
    return results
