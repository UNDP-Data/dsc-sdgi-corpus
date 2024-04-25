# standard library
from collections import Counter
from itertools import combinations
from typing import Literal

# wrangling
import numpy as np
import torch
import dgl


__all__ = [
    "construct_nodes_and_edges",
    "construct_graph",
]


def construct_nodes_and_edges(
    text: str,
    vocabulary: set[str] = None,
    scores: dict[str, float] = None,
    window_size: int = 2,
    directed: bool = False,
    max_nodes: int = 10_000,
    max_edges: int = 10_000,
    edge_strategy: Literal["window", "random", "none"] = "window",
):
    tokens = text.split()
    nodes = Counter(
        [token for token in tokens if vocabulary is None or token in vocabulary]
    )
    nodes = dict(nodes.most_common(max_nodes))
    if scores is None:
        scores = {node: 1 / 2 for node in nodes}

    edges = Counter()
    for i in range(0, len(tokens), window_size):
        if edge_strategy == "window":
            features = tokens[i : i + window_size]
        elif edge_strategy == "random":
            from random import sample

            features = sample(list(nodes), k=window_size)
        elif edge_strategy == "none":
            continue
        else:
            raise ValueError(f"Unexpected value for `edge_strategy`: {edge_strategy}")
        # calculate scores for all pairs of tokens, note that combinations are successive
        for u, v in combinations(features, 2):
            if u not in nodes or v not in nodes:
                continue
            weight = scores[u] + scores[v]
            edges.update({(u, v): weight})
            if not directed:
                # undirected has the same weight in both directions
                edges.update({(v, u): weight})

    edges = dict(edges.most_common(max_edges))
    return nodes, edges


def construct_graph(text: str, vocabulary: list[str], **kwargs):
    token2id = {token: index for index, token in enumerate(vocabulary)}
    nodes, edges = construct_nodes_and_edges(text, vocabulary=set(vocabulary), **kwargs)
    nodes = dict(sorted(nodes.items(), key=lambda x: x[0]))
    mapping = {token: index for index, token in enumerate(nodes)}
    if edges:
        U, V = zip(*[(mapping[u], mapping[v]) for u, v in edges])
    else:
        U, V = [], []
    g = dgl.graph((U, V), num_nodes=len(nodes))

    # convert to 'undirected' graph
    # g = dgl.add_reverse_edges(g)  # no need for this if get_nodes_and_edges uses `directed`

    token_ids = np.asarray([token2id[node] for node in nodes]).reshape(-1, 1)
    g.ndata["attr"] = torch.LongTensor(token_ids)

    weights = np.asarray(list(nodes.values())).reshape(-1, 1)
    weights = np.log(weights) + 1
    g.ndata["w"] = torch.FloatTensor(weights)

    weights = np.asarray(list(edges.values())).reshape(-1, 1)
    weights = np.log(weights) + 1
    g.edata["w"] = torch.FloatTensor(weights)

    # # for nodes with in-degree of 1, may be commented out for the full dataset
    g = dgl.add_self_loop(g)
    return g
