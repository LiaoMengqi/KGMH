import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from tqdm import tqdm

from base_models.transe_base import TransEBase
import utils.data_process as dps
from data.data_loader import DataLoader
import utils.metrics as mtc


class TransE(nn.Module):
    def __init__(self,
                 transe_base: TransEBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(TransE, self).__init__()
        self.model = transe_base
        self.data = data
        self.opt = opt
        self.name='transe'

    def train_epoch(self,
                    batch_size: int) -> float:
        self.train()
        self.opt.zero_grad()
        data_batched = dps.batch_data(self.data.train, batch_size)
        total_batch = int(len(self.data.train) / batch_size) + (len(self.data.train) % batch_size != 0)
        total_loss = 0
        for batch_index in tqdm(range(total_batch)):
            batch = next(data_batched)
            neg_sample = dps.generate_negative_sample(batch, self.data.num_entity)
            loss = self.loss(batch, neg_sample)
            loss.backward()
            self.opt.step()
            total_loss = total_loss + float(loss)
        return total_loss

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False):
        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']
        if dataset == 'valid':
            data = dps.batch_data(self.data.valid, batch_size)
            total_batch = int(len(self.data.valid) / batch_size) + (len(self.data.valid) % batch_size != 0)
        elif dataset == 'test':
            data = dps.batch_data(self.data.test, batch_size)
            total_batch = int(len(self.data.test) / batch_size) + (len(self.data.test) % batch_size != 0)
        else:
            raise Exception

        rank_list = []
        for batch_index in tqdm(range(total_batch)):
            batch = next(data)
            with torch.no_grad():
                hr = self.model.get_entity_embedding(batch[:, 0]) + self.model.get_relation_embedding(batch[:, 1])
                t = self.model.get_entity_embedding(torch.LongTensor(range(self.data.num_entity)).to(hr.device))
                score = -((hr.unsqueeze(1) - t.unsqueeze(0)).norm(p=self.model.p_norm, dim=-1))
                score = score.cpu().numpy()
                rank = mtc.calculate_rank(score, batch[:, 1].cpu().numpy())
                rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_rank)
        return metrics

    def loss(self,
             pos_edge,
             nag_edge):
        # self.model.norm_weight()
        pos_dis = self.model(pos_edge)
        nag_dis = self.model(nag_edge)
        loss = F.relu(self.model.margin + pos_dis - nag_dis).sum()
        # scale loss
        relation = torch.unique(pos_edge[:, 1])
        rela_scale_loss = self.scale_loss(self.model.relation_embedding(relation))
        entity = torch.unique(torch.cat([pos_edge[:, 0], pos_edge[:, 2], nag_edge[:, 0], nag_edge[:, 2]]))
        entity_scale_loss = self.scale_loss(self.model.entity_embedding(entity))
        # compose loss
        loss = loss + self.model.c_e * entity_scale_loss + self.model.c_r * rela_scale_loss
        return loss

    def scale_loss(self, embed):
        return F.relu(embed.norm(p=2, dim=-1) - 1).sum() / embed.shape[0]

    def get_config(self):
        config = {}
        config['model'] = self.name
        config['dataset'] = self.data.dataset
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['emb_dim'] = self.model.emb_dim
        config['margin'] = self.model.margin
        config['p_norm'] = self.model.p_norm
        config['c_e'] = self.model.c_e
        config['c_r'] = self.model.c_r
        return config
