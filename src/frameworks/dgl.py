# standard library
from typing import Literal

# wrangling
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import SAGEConv

# evaluations
from sklearn.metrics import f1_score

# utils
from tqdm import tqdm

# local packages
from ..preprocessing import construct_graph
from ..evaluation import predict_labels

__all__ = [
    "MultilabelTextDataset",
    "SAGEGraph",
    "train_model",
]


class MultilabelTextDataset(DGLDataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[list[int]],
        vocabulary: list[str],
        directed: bool = True,
    ):
        super().__init__(name="MultilabelTextDataset")
        self.vocabulary = vocabulary
        self.directed = directed
        self.construct_graphs(texts)
        self.labels = torch.LongTensor(np.vstack(labels))

    def construct_graphs(self, texts: list[str]):
        self.graphs = []
        for text in tqdm(texts):
            g = construct_graph(
                text=text,
                vocabulary=self.vocabulary,
                window_size=2,
                directed=self.directed,
                edge_strategy="window",
            )
            self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class SAGEGraph(nn.Module):
    def __init__(self, vocab_size: int, in_feats: int, h_feats: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, in_feats, max_norm=1.0)
        self.dropout = nn.Dropout(p=0.4)
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")

    def forward(self, g):
        h = self.embed(g.ndata["attr"]).squeeze() * g.ndata["w"]
        h = self.dropout(h)
        h = self.conv1(g, h, edge_weight=g.edata["w"])
        h = self.conv2(g, h, edge_weight=g.edata["w"])
        g.ndata["h"] = h
        output = dgl.mean_nodes(g, "h", weight="w")
        return output


def train_model(
    model,
    optimizer,
    epochs: int,
    train_dataloader,
    valid_dataloader=None,
    class_weights: np.ndarray = None,
    patience: int = 5,
    device: Literal["cpu", "cuda"] = "cpu",
):
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
    with tqdm(range(epochs)) as t:
        losses = list()
        score_valid_best = -np.inf
        epoch_valid_best = 0
        score = 0.0
        model = model.to(device)
        for epoch in t:
            model.train()
            for g, labels in train_dataloader:
                g = g.to(device)
                labels = labels.to(device)
                pred = model(g)
                loss = F.binary_cross_entropy_with_logits(
                    pred, labels.float(), weight=class_weights
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy().item())

            # validation and early stopping
            if valid_dataloader is not None:
                with torch.no_grad():
                    y_valid, y_pred = predict_labels(
                        model, valid_dataloader, device=device
                    )
                score = f1_score(y_valid, y_pred, average="macro", zero_division=0.0)
                if score >= score_valid_best:
                    score_valid_best = score
                    epoch_valid_best = epoch
                else:
                    if epoch - epoch_valid_best > patience:
                        print(
                            f"Early stopping epoch {epoch} | Best F-1 score {score:.2f}"
                        )
                        break
            t.set_description(
                f"Loss: {np.mean(losses):.3f} Validation score: {score:.3f}"
            )
    model = model.eval().cpu()
    return model
