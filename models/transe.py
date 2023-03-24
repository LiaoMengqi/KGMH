import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from tqdm import tqdm

from base_models.transe_base import TransEBase
import utils.data_process as dps
from data_loader import DataLoader
import utils.metrics as mtc


class TransE(nn.Module):
    def __init__(self, transe_base: TransEBase, data: DataLoader, opt: torch.optim.Optimizer):
        super(TransE, self).__init__()
        self.model = transe_base
        self.data = data
        self.opt = opt

    def train_epoch(self, batch_size: int) -> float:
        data_batched = dps.batch_data(self.data.train, batch_size)
        total_loss = 0
        for batch in tqdm(data_batched):
            self.model.train()
            self.model.zero_grad()
            neg_sample = dps.generate_negative_sample(batch, self.data.num_entity)
            loss = self.loss(batch, neg_sample)
            loss.backward()
            self.opt.step()
            total_loss = total_loss + float(loss)
        return total_loss

    def test(self, batch_size, dataset='valid', mode='obj', metric_list=None):
        if metric_list is None:
            metric_list = ['hist@1', 'hist@3', 'hist@10']
        if dataset == 'valid':
            data_batched = dps.batch_data(self.data.valid, batch_size)
        elif dataset == 'test':
            data_batched = dps.batch_data(self.data.test, batch_size)
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')
        rank_list = []
        for batch in tqdm(data_batched):
            with torch.no_grad():
                hr = self.model.get_entity_embedding(batch[:, 0]) + self.model.get_relation_embedding(batch[:, 1])
                t = self.model.get_entity_embedding(torch.LongTensor(range(self.data.num_entity)).to(hr.device))
                score = -((hr.unsqueeze(1) - t.unsqueeze(0)).norm(p=self.model.p_norm, dim=-1))
                score = score.cpu().numpy()
                rank = mtc.calculate_rank(score, batch[:, 1].cpu().numpy())
                rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        metric_dict = {}
        for metric in metric_list:
            if re.match(r'hist@\d+', metric):
                n = int(re.findall(r'\d+', metric)[0])
                metric_dict[metric] = mtc.calculate_hist(n, all_rank)
            elif metric == 'mr':
                metric_dict['mr'] = all_rank.mean()
        return metric_dict

    def loss(self, pos_edge, nag_edge):
        self.model.norm_weight()
        pos_dis = self.model(pos_edge)
        nag_dis = self.model(nag_edge)
        loss = F.relu(self.model.margin + pos_dis - nag_dis).sum()
        '''
        rela_scale_loss = self.scale_loss(self.model.relation_embedding(pos_edge[:, 1]))
        entity = torch.cat([pos_edge[:, 0], pos_edge[:, 2], nag_edge[:, 0], nag_edge[:, 2]])
        entity_scale_loss = self.scale_loss(self.model.entity_embedding(entity))
        loss = loss + self.model.c * (rela_scale_loss + entity_scale_loss)
        '''
        return loss

    def scale_loss(self, embed):
        return F.relu(embed.norm(p=2, dim=1) - 1).sum() / embed.shape[0]
