import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from typing import Dict, List

def _calculate_ndcg_at_k(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray, k: int) -> float:
    y_true_reshaped = np.expand_dims(y_true_relevance, axis=0)
    y_pred_reshaped = np.expand_dims(y_pred_scores, axis=0)
    return ndcg_score(y_true_reshaped, y_pred_reshaped, k=k)

def _calculate_precision_at_k(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray, k: int) -> float:
    top_k_pred_indices = np.argsort(y_pred_scores)[::-1][:k]
    top_k_true_indices = np.argsort(y_true_relevance)[::-1][:k]
    hits = len(set(top_k_pred_indices) & set(top_k_true_indices))
    return hits / k

def _calculate_spearman_correlation(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray) -> float:
    correlation, _ = spearmanr(y_true_relevance, y_pred_scores)
    return correlation

def evaluate_ranking_model(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray, top_k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
    y_true_relevance = np.max(y_true_ranks) - y_true_ranks
    results = {}
    results['spearman_correlation'] = _calculate_spearman_correlation(y_true_relevance, y_pred_scores)
    for k in top_k_list:
        results[f'ndcg@{k}'] = _calculate_ndcg_at_k(y_true_relevance, y_pred_scores, k=k)
        results[f'precision@{k}'] = _calculate_precision_at_k(y_true_relevance, y_pred_scores, k=k)
    return results