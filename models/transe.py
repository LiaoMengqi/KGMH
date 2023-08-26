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
from models.mate_model import MateModel


class TransE(MateModel):
    def __init__(self,
                 transe_base: TransEBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(TransE, self).__init__()
        self.model = transe_base
        self.data = data
        self.opt = opt
        self.name = 'transe'
        self.ans = None

    def train_epoch(self,
                    batch_size=128):
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
            total_loss += float(loss)
        return total_loss / total_batch

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False):
        if filter_out and self.ans is None:
            self.ans = dps.get_answer(torch.cat([self.data.train, self.data.valid, self.data.test], dim=0),
                                 self.data.num_entity, self.data.num_relation)
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
        rank_list_filter = []
        for batch_index in tqdm(range(total_batch)):
            batch = next(data)
            with torch.no_grad():
                hr = self.model.get_entity_embedding(batch[:, 0]) + self.model.get_relation_embedding(batch[:, 1])
                t = self.model.get_entity_embedding(torch.LongTensor(range(self.data.num_entity)).to(hr.device))
                score = -((hr.unsqueeze(1) - t.unsqueeze(0)).norm(p=self.model.p_norm, dim=-1))
                rank = mtc.calculate_rank(score, batch[:, 1])
                rank_list.append(rank)
                if filter_out:
                    score = dps.filter_score(score, self.ans, batch, self.data.num_relation)
                    rank = mtc.calculate_rank(score, batch[:, 2])
                    rank_list_filter.append(rank)
        all_rank = torch.cat(rank_list, dim=-1)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_rank)
        if filter_out:
            all_rank = torch.cat(rank_list_filter)
            metrics_filter = mtc.ranks_to_metrics(metric_list, all_rank, filter_out)
            metrics.update(metrics_filter)
        return metrics

    def loss(self,
             pos_edge,
             nag_edge):
        # self.model.norm_weight()
        pos_dis = self.model(pos_edge)
        nag_dis = self.model(nag_edge)
        """loss = (torch.max(pos_dis - nag_dis,
                          -torch.Tensor([self.model.margin]).cuda())).mean()  # + self.margin"""
        loss = F.relu(self.model.margin + pos_dis - nag_dis).mean()
        # margin = -torch.Tensor([self.model.margin, ]).to(pos_dis.device)
        # loss = (torch.max(pos_dis - nag_dis, margin)).mean()
        # scale loss
        scale_loss = 0
        if self.model.c_r != 0:
            relation = torch.cat([pos_edge[:, 1], nag_edge[:, 1]])
            rela_scale_loss = self.scale_loss(self.model.relation_embedding(relation))
            scale_loss = scale_loss + self.model.c_r * rela_scale_loss
        if self.model.c_e != 0:
            entity = torch.cat([pos_edge[:, 0], pos_edge[:, 2], nag_edge[:, 0], nag_edge[:, 2]])
            entity_scale_loss = self.scale_loss(self.model.entity_embedding(entity))
            scale_loss = scale_loss + self.model.c_e * entity_scale_loss
        # compose loss
        loss = loss + scale_loss
        return loss

    def scale_loss(self, embed):
        return F.relu(embed.norm(p=2, dim=-1) - 1).mean()

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
