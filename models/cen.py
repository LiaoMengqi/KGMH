import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from base_models.cen_base import CENBase
from data.data_loader import DataLoader
import utils.data_process as dps
import utils.metrics as mtc
import utils
from models.mate_model import MateModel


class CEN(MateModel):
    def __init__(self,
                 model: CENBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(CEN, self).__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.name = 'cen'

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
             metric_list=None,
             filter_out=False):
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
        rank_list_filter = []
        with torch.no_grad():
            for edge in tqdm(data):
                score = self.model.forward(history, edge, training=False)
                ranks = mtc.calculate_rank(score, edge[:, 2])
                rank_list.append(ranks)
                if filter_out:
                    ans = utils.data_process.get_answer(edge, self.data.num_entity, self.data.num_relation * 2)
                    score = utils.data_process.filter_score(score, ans, edge, self.data.num_relation * 2)
                    rank = mtc.calculate_rank(score, edge[:, 2])
                    rank_list_filter.append(rank)
        all_ranks = torch.cat(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_ranks)
        if filter_out:
            all_rank = np.concatenate(rank_list_filter)
            metrics_filter = mtc.ranks_to_metrics(metric_list, all_rank, filter_out)
            metrics.update(metrics_filter)
        return metrics

    def get_config(self):
        config = {}
        config['model'] = 'cen'
        config['dataset'] = self.data.dataset
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['dim'] = self.model.dim
        config['dropout'] = self.model.dropout_value
        config['channel'] = self.model.channel
        config['width'] = self.model.width
        config['seq_len'] = self.model.seq_len
        config['layer_norm'] = self.model.layer_norm
        return config
