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
                 opt: torch.optim.Optimizer,
                 mode='prediction',
                 w=5,
                 ):
        super(RGCN, self).__init__()
        self.model = rgcn_base
        self.data = data
        self.opt = opt
        self.mode = mode
        if self.mode == 'prediction':
            self.w = w
            self.dist_mult = DistMult(self.model.num_relation, self.model.dims[-1])
            self.opt.add_param_group({'params': self.dist_mult.parameters()})
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')

        self.h = None

    def train_epoch(self,
                    batch_size: int,
                    save_h=False,
                    ):
        self.train()
        self.opt.zero_grad()
        edge = self.data.train
        total_loss = 0
        if self.mode == 'prediction':
            # generate negative edge
            nag_list = []
            for i in range(self.w):
                nag_list.append(generate_negative_sample(edge, self.model.num_entity))
            nag_edge = torch.cat(nag_list, dim=0)
            data = torch.cat([edge, nag_edge], dim=0)
            target = torch.cat([torch.ones(edge.shape[0], dtype=torch.long, device=data.device),
                                torch.zeros(nag_edge.shape[0], dtype=torch.long, device=data.device)]).unsqueeze(1)
            data = torch.cat([data, target], dim=-1)
            batches = dps.batch_data(data, batch_size)
            h_output = self.model.forward(edge)
            for batch in tqdm(batches):
                loss = self.loss(h_output, edge=batch[:, 0:3], link_tag=batch[:, 3])
                loss.backward(retain_graph=True)
                total_loss = total_loss + float(loss)
            if save_h:
                self.h = h_output.detach().clone()
        else:
            raise NotImplementedError
        # full batch optimization
        self.opt.step()
        return total_loss

    def test(self, batch_size,
             dataset='valid',
             metric_list=None) -> dict:
        if metric_list is None:
            metric_list = ['hist@1', 'hist@3', 'hist@10']
        if self.h is None:
            with torch.no_grad():
                self.h = self.model.forward(self.data.train)
        if dataset == 'valid':
            data_batched = dps.batch_data(self.data.valid, batch_size)
        elif dataset == 'test':
            data_batched = dps.batch_data(self.data.test, batch_size)
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')
        rank_list = []
        for batch in tqdm(data_batched):
            with torch.no_grad():
                sr = batch[:, :2]
                obj = torch.arange(0, self.h.shape[0], device=sr.device).reshape(1, self.h.shape[0], 1)
                obj = obj.expand(sr.shape[0], -1, 1)
                sr = sr.unsqueeze(1).expand(-1, self.h.shape[0], 2)
                edges = torch.cat((sr, obj), dim=2)
                score = self.dist_mult(self.h[edges[:, :, 0]], self.h[edges[:, :, 2]], edges[:, :, 1])
                rank = mtc.calculate_rank(score.cpu().numpy(), batch[:, 1].cpu().numpy())
                rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list, all_rank)
        return metrics

    def weight_decay(self, penalty=0.01):
        return self.dist_mult.diag.pow(2).sum() * penalty

    def loss(self,
             h_input,
             edge=None,
             link_tag=None,
             class_tag=None,
             use_l2_regularization=False) -> torch.Tensor:
        if self.mode == 'classification':
            raise NotImplementedError
        elif self.mode == 'prediction':
            score = nn.functional.sigmoid(self.dist_mult.forward(h_input[edge[:, 0]],
                                                                 h_input[edge[:, 2]],
                                                                 edge[:, 1])).unsqueeze(1)
            score = torch.cat([1 - score, score], dim=1)
            loss = self.cross_entropy(score, link_tag) / edge.shape[0]
            if use_l2_regularization:
                loss.add_(self.weight_decay())
        else:
            raise NotImplementedError
        return loss
