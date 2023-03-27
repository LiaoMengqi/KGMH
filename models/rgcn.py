import torch
import torch.nn as nn
import numpy as np

from base_models.regcn_base import RGCNBase
from data.data_loader import DataLoader
from base_models.distmilt_base import DistMult
from utils.data_process import generate_negative_sample
import utils.data_process as dps
from tqdm import tqdm
import utils.metrics as mtc


class RGCN(nn.Module):
    def __init__(self, rgcn_base: RGCNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer,
                 mode='prediction',
                 w=5):
        super(RGCN, self).__init__()
        self.model = rgcn_base
        self.data = data
        self.opt = opt
        self.mode = mode
        if self.mode == 'prediction':
            self.w = w
            self.dist_mult = DistMult(self.model.num_relation, self.model.dims[-1])
            opt.add_param_group({'params': self.dist_mult.parameters()})
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.h = None

    def train_epoch(self,
                    save_h=False):
        self.model.train()
        self.opt.zero_grad()
        edge = self.data.train
        if self.mode == 'prediction':
            nag_list = []
            for i in range(self.w):
                nag_list.append(generate_negative_sample(edge, self.model.num_entity))
            nag_edge = torch.cat(nag_list, dim=0)
            h_output = self.model.forward(edge)
            if save_h:
                self.h = h_output.detach().clone()
            loss = self.loss(h_output, edge, nag_edge)
        else:
            loss = None
        loss.backward()
        self.opt.step()
        return float(loss)

    def test(self, batch_size,
             dataset='valid',
             metric_list=None) -> dict:
        if metric_list is None:
            metric_list = ['hist@1', 'hist@3', 'hist@10']
        if self.h is None:
            raise Exception
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
                obj = torch.arange(0, self.h.shape[0],device=sr.device).reshape(1, self.h.shape[0], 1)
                obj = obj.expand(sr.shape[0], -1, 1)
                sr = sr.unsqueeze(1).expand(-1, self.h.shape[0], 2)
                edges = torch.cat((sr, obj), dim=2)
                score = self.dist_mult(self.h[edges[:, :, 0]], self.h[edges[:, :, 2]], edges[:, :, 1])
                rank = mtc.calculate_rank(score.cpu().numpy(), batch[:, 1].cpu().numpy())
                rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list, all_rank)
        return metrics

    def loss(self, h_input,
             pos_edge=None,
             nag_edge=None,
             tag=None) -> torch.Tensor:
        if self.mode == 'classification':
            loss = 1
        elif self.mode == 'prediction':
            sub_embed_pos = h_input[pos_edge[:, 0]]
            obj_embed_pos = h_input[pos_edge[:, 2]]
            sub_embed_nag = h_input[nag_edge[:, 0]]
            obj_embed_nag = h_input[nag_edge[:, 2]]

            pos_score = nn.functional.sigmoid(self.dist_mult.forward(sub_embed_pos, obj_embed_pos, pos_edge[:, 1]))
            pos_score = pos_score.unsqueeze(1)
            pos_score = torch.cat([1 - pos_score, pos_score], dim=1)
            target = torch.ones(pos_score.shape[0], dtype=torch.long, device=pos_score.device)
            loss = self.cross_entropy(pos_score, target)

            nag_score = nn.functional.sigmoid(self.dist_mult.forward(sub_embed_nag, obj_embed_nag, nag_edge[:, 1]))
            nag_score = nag_score.unsqueeze(1)
            nag_score = torch.cat([1 - nag_score, nag_score], dim=1)
            target = torch.zeros(pos_score.shape[0], dtype=torch.long, device=pos_score.device)
            loss = loss + self.cross_entropy(pos_score, target)

            loss = loss / (pos_score.shape[0] + nag_score.shape[0])
        else:
            raise Exception('mode error!')
        return loss
