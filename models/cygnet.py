from base_models.cygnet_base import CyGNetBase
from data.data_loader import DataLoader
import torch.nn as nn
import torch
import torch.sparse as sp
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc
import numpy as np


class CyGNet(nn.Module):
    def __init__(self, model: CyGNetBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer
                 ):
        super(CyGNet, self).__init__()
        self.data = data
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.opt = opt
        self.model = model
        self.vocabulary = torch.sparse_coo_tensor(size=([data.num_entity * data.num_relation, data.num_entity]),
                                                  device=data.train.device)

    def update_vocabulary(self, entity, rela, target):
        i = entity * self.data.num_relation + rela
        i = torch.cat([i.unsqueeze(0), target.unsqueeze(0)], dim=0)
        v = torch.ones(i.shape[-1])
        new_vocabulary = torch.sparse_coo_tensor(i, v, size=self.vocabulary.shape, device=self.vocabulary.device)
        self.vocabulary.add_(new_vocabulary)
        self.vocabulary = torch.sign(self.vocabulary)

    def get_vocabulary(self, entity, rela) -> torch.Tensor:
        i = entity * self.data.num_relation + rela
        vocabulary = torch.index_select(self.vocabulary, index=i, dim=0)
        return vocabulary.to_dense()

    def train_epoch(self, batch_size):
        self.model.train()
        all_loss = 0
        for i in tqdm(range(len(self.train_data))):
            time_stamp = self.train_time[i]
            data_batched = dps.batch_data(self.train_data[i], batch_size=batch_size)
            for batch in data_batched:
                self.opt.zero_grad()
                score = self.model.forward(batch, self.get_vocabulary(batch[:, 0], batch[:, 1]), time_stamp)
                loss = self.loss(torch.log(score), batch[:, 2])
                loss.backward()
                all_loss = all_loss + float(loss)
                self.opt.step()
                # self.model.nan_to_zero()
            self.update_vocabulary(self.train_data[i][:, 0], self.train_data[i][:, 1], self.train_data[i][:, 2])
        return all_loss

    def loss(self, score, label) -> torch.Tensor:
        return torch.nn.functional.nll_loss(score, label) + self.regularization_loss()

    def regularization_loss(self, reg_fact=0.01):
        regularization_loss = torch.mean(self.model.relation_embed.pow(2)) + torch.mean(
            self.model.entity_embed.pow(2)) + torch.mean(self.model.unit_time_embed.pow(2))
        return regularization_loss * reg_fact

    def test(self,
             batch_size,
             dataset='valid',
             mode='obj',
             metric_list=None):
        if metric_list is None:
            metric_list = ['hist@1', 'hist@10', 'hist@100', 'mr']
        if dataset == 'valid':
            data = self.valid_data
            time_list = self.valid_time
        elif dataset == 'test':
            data = self.test_data
            time_list = self.test_time
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')
        rank_list = []
        for i in tqdm(range(len(data))):
            batches = dps.batch_data(data[i], batch_size=batch_size)
            time_stamp = time_list[i]
            for batch in batches:
                with torch.no_grad():
                    score = self.model.forward(batch, self.get_vocabulary(batch[:, 0], batch[:, 1]), time_stamp)
                    rank = mtc.calculate_rank(score.cpu().numpy(), batch[:, 1].cpu().numpy())
                    rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list, all_rank)
        return metrics
