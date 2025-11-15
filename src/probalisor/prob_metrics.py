# %%
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from typing import Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from typing import Dict, List

def evaluate_classifier(
    y_true,
    y_pred_scores,
    threshold: float = 0.5,
    top_k_list: List[int] = None
) -> Dict[str, float]:
    """
    Évalue un classifieur binaire avec un focus sur la précision des prédictions positives.
    
    y_true : array-like (0/1)
    y_pred_scores : array-like (probas ou scores)
    threshold : seuil de décision pour classer en positif
    top_k_list : liste d'entiers pour calculer precision@k
    """
    # Conversion en arrays 1D
    y_true = np.asarray(y_true).ravel()
    y_pred_scores = np.asarray(y_pred_scores).ravel()

    if y_true.shape[0] != y_pred_scores.shape[0]:
        raise ValueError(
            f"Incohérence : y_true contient {y_true.shape[0]} échantillons, "
            f"y_pred_scores en contient {y_pred_scores.shape[0]}"
        )

    # Prédictions binaires
    y_pred_labels = (y_pred_scores >= threshold).astype(int)

    results = {}
    results["precision"] = precision_score(y_true, y_pred_labels, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred_labels, zero_division=0)
    results["average_precision"] = average_precision_score(y_true, y_pred_scores)
    results["roc_auc"] = roc_auc_score(y_true, y_pred_scores)
    
    results['spread'] = np.max(y_pred_scores) - np.min(y_pred_scores)

    # Precision@k si demandé
    if top_k_list is not None:
        sorted_idx = np.argsort(y_pred_scores)[::-1]
        for k in top_k_list:
            if k > len(sorted_idx):
                raise ValueError(f"k={k} est supérieur au nombre d'échantillons ({len(sorted_idx)})")
            top_k_idx = sorted_idx[:k]
            results[f"precision@{k}"] = y_true[top_k_idx].sum() / k

    return results