import numpy as np
from scipy.stats import rankdata


def calculate_rank(scores: np.ndarray, target_index: np.ndarray):
    return rankdata(-scores, axis=-1)[np.arange(target_index.shape[0]), target_index]


def calculate_hist(k: int, ranks: np.ndarray):
    return np.sum(ranks <= k) / len(ranks)


def calculate_mrr(ranks: np.ndarray):
    return (1. / ranks).sum() / ranks.shape[0]


def calculate_mse(output: np.ndarray, label: np.ndarray):
    return np.sum(np.power(output - label, 2)) / output.shape[0]
