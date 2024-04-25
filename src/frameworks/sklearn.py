# modelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

__all__ = [
    "get_vectoriser",
    "get_sgd_classifier",
    "get_mlp_classifier",
]


def get_vectoriser(
    max_features: int = 100_000, ngram_range=(1, 1), stopwords: list[str] = None
):
    vectoriser = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=True,
        stop_words=stopwords,
        ngram_range=ngram_range,
        min_df=10,
        max_df=0.9,
        max_features=max_features,
        binary=True,
    )
    return vectoriser


def get_sgd_classifier():
    clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        shuffle=True,
        learning_rate="optimal",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        class_weight="balanced",
    )
    return clf


def get_mlp_classifier():
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.01,
        max_iter=1000,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )
    return clf


def run_grid_search(pipe, params, x_train, y_train):
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        scoring="f1_macro",
        n_jobs=-1,
        cv=5,
        error_score=0.0,
    )
    grid.fit(x_train, y_train)
    return grid
