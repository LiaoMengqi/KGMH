import torch
import torch.nn as nn
import numpy as np

from base_models.rgcn_base import RGCNBase
from data.data_loader import DataLoader
from base_models.distmilt_base import DistMult
from utils.data_process import generate_negative_sample
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc
from models.mate_model import MateModel


class RGCN(MateModel):
    def __init__(self,
                 rgcn_base: RGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer
                 ):
        super(RGCN, self).__init__()
        self.model = rgcn_base
        self.data = data
        self.opt = opt
        self.name = 'rgcn'

        self.dist_mult = DistMult(self.model.num_relation, self.model.dims[-1])
        self.opt.add_param_group({'params': self.dist_mult.parameters()})
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')

    def train_epoch(self,
                    batch_size=128):
        self.train()
        self.opt.zero_grad()
        edge = self.data.train
        total_loss = 0
        # generate negative edge
        nag_edge = generate_negative_sample(edge, self.model.num_entity)
        data = torch.cat([edge, nag_edge], dim=0)
        target = torch.cat([torch.ones(edge.shape[0], dtype=torch.long, device=data.device),
                            torch.zeros(nag_edge.shape[0], dtype=torch.long, device=data.device)]).unsqueeze(1)
        data = torch.cat([data, target], dim=-1)
        h_output = self.model.forward(edge)

        batches = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)
        for batch_index in tqdm(range(total_batch)):
            batch = next(batches)
            loss = self.loss(h_output, edge=batch[:, 0:3], link_tag=batch[:, 3])
            loss.backward(retain_graph=True)
            self.opt.step()
            total_loss += float(loss)
        # full batch optimization

        return total_loss / total_batch

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False) -> dict:
        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']

        if dataset == 'valid':
            data_batched = dps.batch_data(self.data.valid, batch_size)
            total_batch = int(len(self.data.valid) / batch_size) + (len(self.data.valid) % batch_size != 0)
        elif dataset == 'test':
            data_batched = dps.batch_data(self.data.test, batch_size)
            total_batch = int(len(self.data.test) / batch_size) + (len(self.data.test) % batch_size != 0)
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')

        rank_list = []
        with torch.no_grad():
            h = self.model.forward(self.data.train)
            for batch_index in tqdm(range(total_batch)):
                batch = next(data_batched)
                sr = batch[:, :2]
                obj = torch.arange(0, h.shape[0], device=sr.device).reshape(1, h.shape[0], 1)
                obj = obj.expand(sr.shape[0], -1, 1)
                sr = sr.unsqueeze(1).expand(-1, h.shape[0], 2)
                edges = torch.cat((sr, obj), dim=2)
                score = self.dist_mult(h[edges[:, :, 0]], h[edges[:, :, 2]], edges[:, :, 1])
                rank = mtc.calculate_rank(score, batch[:, 1])
                rank_list.append(rank)
        all_rank = torch.cat(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list, all_rank)
        return metrics

    def weight_decay(self, penalty=0.01):
        return self.dist_mult.diag.pow(2).sum() * penalty

    def loss(self,
             h_input,
             edge=None,
             link_tag=None,
             use_l2_regularization=False) -> torch.Tensor:
        score = nn.functional.sigmoid(self.dist_mult.forward(h_input[edge[:, 0]],
                                                             h_input[edge[:, 2]],
                                                             edge[:, 1])).unsqueeze(1)
        score = torch.cat([1 - score, score], dim=1)
        loss = self.cross_entropy(score, link_tag)
        if use_l2_regularization:
            loss = loss + self.weight_decay()
        return loss

    def get_config(self):
        config = {}
        config['model'] = 'rgcn'
        return config
