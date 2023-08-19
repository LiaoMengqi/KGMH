import numpy as np
import torch
import re


def calculate_rank(scores: torch.Tensor,
                   target_index: torch.Tensor):
    # higher score get higher rank
    device = scores.device
    x = torch.Tensor(scores)
    _, sindex = torch.sort(-x, dim=1)
    res = torch.zeros_like(x, dtype=sindex.dtype, device=device)
    index = torch.arange(x.shape[1], device=device).expand(x.shape)
    t_rank = res.scatter(dim=1, index=sindex, src=index).float() + 1
    return t_rank[torch.arange(scores.shape[0]), target_index]


def calculate_hist(k: int, ranks: torch.Tensor):
    return float((ranks <= k).sum() / len(ranks))


def calculate_mrr(ranks: torch.Tensor):
    return float((1. / ranks).sum() / ranks.shape[0])


def calculate_mr(ranks: torch.Tensor):
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
