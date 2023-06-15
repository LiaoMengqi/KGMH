import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from base_models.cen_base import CENBase
from data.data_loader import DataLoader
import utils.data_process as dps
import utils.metrics as mtc


class CEN(nn.Module):
    def __init__(self,
                 model: CENBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(CEN, self).__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.seq_len = model.seq_len
        self.grad_norm = 1.0
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def train_epoch(self,
                    batch_size=128):
        self.model.train()
        self.opt.zero_grad()
        # add reverse relation to graph
        data = dps.add_reverse_relation(self.train_data, self.data.num_relation)
        # target time for predict
        total_loss = 0
        for i in tqdm(range(len(data))):
            if i == 0:
                # no history data
                continue
            # history data
            if i >= self.seq_len:
                history_graphs = data[i - self.seq_len:i]
            else:
                history_graphs = data[0:i]
            score = self.model.forward(history_graphs,
                                       data[i])
            loss = self.loss(score, data[i][:, 2])
            loss.backward()
            total_loss = total_loss + float(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
            self.opt.step()
        return total_loss

    def loss(self,
             score,
             target):
        return self.cross_entropy_loss(score, target)

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None):
        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']
        if dataset == 'valid':
            data = self.valid_data
            history = self.train_data
        elif dataset == 'test':
            data = self.test_data
            history = self.valid_data
        else:
            raise Exception
        data = dps.add_reverse_relation(data, self.data.num_relation)
        if self.model.seq_len < len(history):
            history = dps.add_reverse_relation(history[-self.model.seq_len:],
                                               self.data.num_relation)
        else:
            history = dps.add_reverse_relation(history,
                                               self.data.num_relation)
        rank_list = []
        with torch.no_grad():
            for edge in tqdm(data):
                score = self.model.forward(history, edge, training=False)
                ranks = mtc.calculate_rank(score.cpu().numpy(), edge[:, 2].cpu().numpy())
                rank_list.append(ranks)
        all_ranks = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_ranks)
        return metrics

    def get_name(self):
        name = 'cen_'
        data = self.data.dataset + '_'
        channel = 'channel' + str(self.model.channel) + '_'
        kernel_width = 'kernel_width' + str(self.model.width) + '_'
        max_seq_len = 'max_seq_len' + str(self.model.seq_len) + '_'
        dim = 'dim' + str(self.model.dim) + '_'
        dropout = 'dropout' + dps.float_to_int_exp(self.model.dropout_value)
        return name + data + channel + kernel_width + max_seq_len + dim + dropout