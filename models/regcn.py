import torch
import torch.nn as nn
from base_models.regcn_base import REGCNBase
from data.data_loader import DataLoader
from base_models.sacn_base import ConvTransEDecoder
import random
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc
import numpy as np
import utils
from models.mate_model import MateModel


class REGCN(MateModel):
    def __init__(self, model: REGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer,
                 ):
        super(REGCN, self).__init__()
        # common parameters
        self.model = model
        self.data = data
        self.opt = opt
        self.name = 'regcn'
        # data process
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)

        self.seq_len = model.seq_len
        self.grad_norm = 1.0

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.decoder = ConvTransEDecoder(model.hidden_dim,
                                         num_channel=50,
                                         kernel_length=3)
        self.opt.add_param_group({'params': self.decoder.parameters()})

    def train_epoch(self, batch_size=512):
        self.train()
        self.opt.zero_grad()
        # add reverse relation to graph
        data = dps.add_reverse_relation(self.train_data, self.data.num_relation)
        # target time for predict
        index = list(range(len(data)))
        random.shuffle(index)
        total_loss = 0
        for i in tqdm(index):
            if i == 0:
                # no history data
                continue
            # history data
            if i >= self.seq_len:
                edges = data[i - self.seq_len:i]
            else:
                edges = data[0:i]
            evolved_entity_embed, evolved_relation_embed = self.model.forward(edges)
            # put into a decoder to calculate score for object
            score = self.decoder(evolved_entity_embed, evolved_relation_embed, data[i][:, :2])
            # calculate loss
            loss = self.loss(score, data[i][:, 2])
            loss.backward()
            total_loss = total_loss + float(loss)
            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
            self.opt.step()
        return total_loss

    def test(self,
             batch_size=512,
             dataset='valid',
             filter_out=False,
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
        rank_list_filter = []
        self.eval()
        with torch.no_grad():
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history)
            for edge in tqdm(data):
                score = self.decoder(evolved_entity_embed, evolved_relation_embed, edge[:, [0, 1]])
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

    def loss(self, score, target):
        return self.cross_entropy_loss(score, target)

    def get_config(self):
        config = {}
        config['model'] = 'regcn'
        config['dataset'] = self.data.dataset
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['hidden_dim'] = self.model.hidden_dim
        config['seq_len'] = self.model.seq_len
        config['num_layer'] = self.model.num_layer
        config['dropout'] = self.model.dropout_value
        config['active'] = self.model.if_active
        config['self_loop'] = self.model.if_self_loop
        config['layer_norm'] = self.model.layer_norm
        return config
