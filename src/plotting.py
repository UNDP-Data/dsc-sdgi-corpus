# standard library
from typing import Literal

# wrangling
import numpy as np
import pandas as pd
import networkx as nx
from datasets import Dataset
from sklearn.metrics import confusion_matrix

# visualisation
import plotly.express as px
import matplotlib.pyplot as plt

# local packages
from .utils import GOALS, COLOURS


def plot_label_distribution(dataset: Dataset):
    df = dataset.to_pandas()
    fig = px.bar(
        data_frame=df.explode("labels").groupby("labels", as_index=False).size(),
        x="labels",
        y="size",
        labels={
            "labels": "Label",
            "size": "Number of Examples",
        },
        title="Distribution of Examples by Label in the Dataset",
    )
    fig.update_traces(marker_color=COLOURS)
    fig.update_layout(xaxis={"type": "category"})
    return fig


def plot_confusion_matrix(y_true, y_pred, height: int = 800):
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")
    df_confusion = pd.DataFrame(matrix, index=GOALS, columns=GOALS)
    fig = px.imshow(
        img=df_confusion,
        text_auto=".1%",
        title="Confusion Matrix (Row-Normalised)",
        height=height,
    )
    fig.update_layout(
        xaxis={"title": "Predicted Label"},
        yaxis={"title": "True Label"},
    )
    return fig


def plot_predictions(probs: list[float], min_prob: float = 0.5):
    probs = np.asarray(probs)
    colours = [
        COLOURS[index] if prob >= min_prob or index == probs.argmax() else "black"
        for index, prob in enumerate(probs)
    ]
    fig = px.bar(
        x=GOALS,
        y=probs.tolist(),
        labels={"x": "Label", "y": "Predicted Probability"},
        title="Predicted Distribution Over Lables",
    )
    fig.update_layout(yaxis={"range": [0, 1]})
    fig.update_traces(marker_color=colours)
    fig.add_hline(y=0.5, line_dash="dash")
    return fig


def plot_graph(
    G: nx.Graph,
    layout: Literal["circular", "kamada", "spring"] = "circular",
    figsize: tuple[int, int] = (15, 15),
):
    fig = plt.figure(figsize=figsize)

    # create a layout
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G)
    else:
        raise ValueError(f"Unknown layout {layout}.")

    # calculate normalised weights for edge line width
    weights = np.asarray(list(nx.get_edge_attributes(G, "weight").values()))
    weights = weights / weights.max() * 2  # heuristic scale value

    # calculate normalised size for node size
    size = np.asarray(list(dict(G.degree()).values()))
    size = size / size.max() * 300  # heuristic scale value

    nx.draw(
        G=G,
        pos=pos,
        node_size=list(size),
        # node_color=PALLETE_GRAYS[1],
        # edge_color=PALLETE_GRAYS[2],
        # font_color=COLOR_PRIMARY,
        width=list(weights),
        with_labels=True,
    )
    return fig
