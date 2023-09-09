import torch
import torch.nn as nn

from base_models.rgcn_base import RGCNBase
from data.data_loader import DataLoader
from base_models.distmult_base import DistMultDecoder
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
        # decoder
        num_rela_expand = self.model.num_relation * 2 if self.model.inverse else self.model.num_relation
        self.dist_mult = DistMultDecoder(num_rela_expand, self.model.dims[-1])
        self.opt.add_param_group({'params': self.dist_mult.parameters()})

        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.ans = None

    def train_epoch(self,
                    batch_size=128):
        self.train()
        self.opt.zero_grad()
        if self.model.inverse:
            edge = dps.add_inverse(self.data.train, self.model.num_relation)
        else:
            edge = self.data.train
        total_loss = 0
        # generate negative edge
        nag_edge = generate_negative_sample(edge, self.model.num_entity)
        data = torch.cat([edge, nag_edge], dim=0)
        target = torch.cat([torch.ones(edge.shape[0], dtype=torch.long, device=data.device),
                            torch.zeros(nag_edge.shape[0], dtype=torch.long, device=data.device)]).unsqueeze(1)
        data = torch.cat([data, target], dim=-1)

        batches = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)
        h_output = self.model.forward(edge)
        for batch_index in tqdm(range(total_batch)):
            batch = next(batches)
            loss = self.loss(h_output, edge=batch[:, 0:3], link_tag=batch[:, 3])
            loss.backward(retain_graph=True)
            # loss.backward()
            total_loss += float(loss)
        self.opt.step()
        return total_loss / total_batch

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False) -> dict:
        if filter_out and self.ans is None:
            self.ans = dps.get_answer(torch.cat([self.data.train, self.data.valid, self.data.test], dim=0),
                                      self.data.num_entity, self.data.num_relation)

        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']

        if dataset == 'valid':
            data = self.data.valid
        elif dataset == 'test':
            data = self.data.test
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')
        if self.model.inverse:
            data = dps.add_inverse(data, self.model.num_relation)
        data_batched = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)

        rank_list = []
        rank_list_filter = []
        self.eval()
        with torch.no_grad():

            if self.model.inverse:
                h = self.model.forward(dps.add_inverse(self.data.train, self.model.num_relation))
            else:
                h = self.model.forward(self.data.train)

            for batch_index in tqdm(range(total_batch)):
                batch = next(data_batched)
                sr = batch[:, :2]
                obj = torch.arange(0, h.shape[0], device=sr.device).reshape(1, h.shape[0], 1)
                obj = obj.expand(sr.shape[0], -1, 1)
                sr = sr.unsqueeze(1).expand(-1, h.shape[0], 2)
                edges = torch.cat((sr, obj), dim=2)
                score = self.dist_mult(h[edges[:, :, 0]], h[edges[:, :, 2]], edges[:, :, 1])
                rank = mtc.calculate_rank(score, batch[:, 2])
                rank_list.append(rank)
                if filter_out:
                    score = dps.filter_score(score, self.ans, batch, self.data.num_relation)
                    rank = mtc.calculate_rank(score, batch[:, 2])
                    rank_list_filter.append(rank)
        all_rank = torch.cat(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list, all_rank)
        if filter_out:
            all_rank = torch.cat(rank_list_filter)
            metrics_filter = mtc.ranks_to_metrics(metric_list, all_rank, filter_out)
            metrics.update(metrics_filter)
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
        config['model'] = self.name
        config['dataset'] = self.data.dataset
        config['dim_list'] = self.model.dims
        config['num_relation'] = self.data.num_relation
        config['num_entity'] = self.data.num_entity
        config['use_basis'] = self.model.use_basis
        config['num_basis'] = self.model.num_basis
        config['use_block'] = self.model.use_block
        config['num_block'] = self.model.num_block
        config['dropout_s'] = self.model.dropout_s
        config['dropout_o'] = self.model.dropout_o
        return config
