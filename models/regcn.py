import torch
import torch.nn as nn
from base_models.regcn_base import REGCNBase
from data.data_loader import DataLoader
from base_models.sacn_base import ConvTransEBase
import random
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc
import numpy as np


class REGCN(nn.Module):
    def __init__(self, model: REGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer,
                 ):
        super(REGCN, self).__init__()
        self.model = model
        self.data = data
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.opt = opt
        self.seq_len = model.seq_len
        self.grad_norm = 1.0

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.decoder = ConvTransEBase(model.hidden_dim,
                                      num_channel=50,
                                      kernel_length=3)
        self.opt.add_param_group({'params': self.decoder.parameters()})

    def train_epoch(self, batch_size=512):
        self.model.train()
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
            evolved_entity_embed, evolved_relation_embed = self.model.forward(history)
            for edge in tqdm(data):
                score = self.decoder(evolved_entity_embed, evolved_relation_embed, edge[:, [0, 1]])
                ranks = mtc.calculate_rank(score.cpu().numpy(), edge[:, 2].cpu().numpy())
                rank_list.append(ranks)
        all_ranks = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_ranks)
        return metrics

    def loss(self, score, target):
        return self.cross_entropy_loss(score, target)

    def get_name(self):
        name = 'regcn_'
        dataset = self.data.dataset + '_'
        dim = 'dim' + str(self.model.hidden_dim) + '_'
        seq_len = 'seq_len' + str(self.model.seq_len) + '_'
        layer = 'layer' + str(self.model.num_layer) + '_'
        dropout = 'dropout' + dps.float_to_int_exp(self.model.dropout_value)
        return name + dataset + dim + seq_len + layer + dropout
