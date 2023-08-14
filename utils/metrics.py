import numpy as np
from scipy.stats import rankdata
import re


def calculate_rank(scores: np.ndarray,
                   target_index: np.ndarray):
    rank = rankdata(-scores, axis=-1)
    return rank[np.arange(scores.shape[0]), target_index]


def calculate_hist(k: int, ranks: np.ndarray):
    return float(np.sum(ranks <= k) / len(ranks))


def calculate_mrr(ranks: np.ndarray):
    return float((1. / ranks).sum() / ranks.shape[0])


def calculate_mr(ranks: np.ndarray):
    return float(ranks.mean())


def ranks_to_metrics(metric_list: list,
                     ranks,
                     filter_out=False):
    metrics = {}
    prefix = ""
    if filter_out:
        prefix = "filter "
    for metric in metric_list:
        if re.match(r'hits@\d+', metric):
            n = int(re.findall(r'\d+', metric)[0])
            metrics[prefix + metric] = calculate_hist(n, ranks)
        elif metric == 'mr':
            metrics[prefix + 'mr'] = calculate_mr(ranks)
        elif metric == 'mrr':
            metrics[prefix + 'mrr'] = calculate_mrr(ranks)
    return metrics


def calculate_mse(output: np.ndarray,
                  label: np.ndarray):
    return np.sum(np.power(output - label, 2)) / output.shape[0]
