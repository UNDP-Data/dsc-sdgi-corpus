import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from .utils import GOALS


def predict_labels(model, dataloader, threshold: float = 0.5, device: str = "cpu"):
    model = model.eval().to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for g, labels in dataloader:
            g = g.to(device)
            labels = labels.to(device)
            predictions = model(g)
            labels = labels.float().detach().cpu().numpy()
            predictions = (
                (F.sigmoid(predictions) >= threshold).float().detach().cpu().numpy()
            )
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
        return y_true, y_pred


def evaluate_results(y_true, y_pred, name: str) -> dict:
    print(f"{name} set:")
    report = classification_report(
        y_true=y_true, y_pred=y_pred, target_names=GOALS, zero_division=0.0
    )
    print(report)

    results = {}
    for average in ("micro", "macro", "weighted"):
        results[f"f1-{average}"] = f1_score(
            y_true=y_true, y_pred=y_pred, average=average, zero_division=0.0
        )
    return results


def evaluate_model(model, x, y, name: str = "") -> dict:
    y_pred = model.predict(x)
    results = evaluate_results(y_true=y, y_pred=y_pred, name=name)
    return results


def evaluate_gnn(model, dataloader, name: str = "") -> dict:
    y_true, y_pred = predict_labels(model, dataloader)
    results = evaluate_results(y_true=y_true, y_pred=y_pred, name=name)
    return results
