import numpy as np
from sklearn.metrics import fbeta_score, average_precision_score

def compute_fbeta(y_true, y_pred, beta: float) -> float:
    """
    Compute F-beta score for binary classification.
    """
    return fbeta_score(y_true, y_pred, beta=beta)


def compute_pr_auc(y_true, y_prob) -> float:
    """
    Compute Precision-Recall AUC (Average Precision).
    """
    return average_precision_score(y_true, y_prob)

