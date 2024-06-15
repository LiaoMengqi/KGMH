from base_models.ggecddcd_base import GGEcdDcdBase
from data.data_loader import DataLoader
from models.mate_model import MateModel
import utils.data_process as dps
import utils.metrics as mtc

import torch
from tqdm import tqdm
import torch.nn as nn


class GGEcdDcd(MateModel):
    def __init__(self,
                 base_model: GGEcdDcdBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer
                 ):
        super(GGEcdDcd, self).__init__()
        self.model = base_model
        self.data = data
        self.opt = opt
        self.name = 'ggecddcd'
        self.ans = None

        self.graph = self.data.train

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')

    def train_epoch(self,
                    batch_size=128):
        edge = self.graph

        total_loss = 0
        # generate negative edge
        nag_edge = dps.generate_negative_sample(edge, self.data.num_entity)
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

    def loss(self,
             h_input,
             edge=None,
             link_tag=None,
             use_l2_regularization=False) -> torch.Tensor:
        if self.model.decoder == 'distmult':
            score = nn.functional.sigmoid(self.model.Decoder.forward(h_input[edge[:, 0]],
                                                                     h_input[edge[:, 2]],
                                                                     edge[:, 1])).unsqueeze(1)
        elif self.model.decoder == 'transe':
            dist = (h_input[edge[:, 0]] + self.model.rela_embed.weight[edge[:, 1]]
                    - h_input[edge[:, 2]]).norm(p=2, dim=-1)
            score = 1 - nn.functional.sigmoid(dist).unsqueeze(1)

        score = torch.cat([1 - score, score], dim=1)
        loss = self.cross_entropy(score, link_tag)
        if use_l2_regularization:
            loss = loss + self.weight_decay()
        return loss

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
            data = self.data.valid
        elif dataset == 'test':
            data = self.data.test
        else:
            raise Exception('dataset ' + dataset + ' is not defined!')
        data_batched = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)

        rank_list = []
        rank_list_filter = []
        self.eval()
        with torch.no_grad():

            h = self.model.forward(self.graph)

            for batch_index in tqdm(range(total_batch)):
                batch = next(data_batched)
                sr = batch[:, :2]
                obj = torch.arange(0, h.shape[0], device=sr.device).reshape(1, h.shape[0], 1)
                obj = obj.expand(sr.shape[0], -1, 1)
                sr = sr.unsqueeze(1).expand(-1, h.shape[0], 2)
                edges = torch.cat((sr, obj), dim=2)
                if self.model.decoder == 'distmult':
                    score = self.model.Decoder(h[edges[:, :, 0]], h[edges[:, :, 2]], edges[:, :, 1])
                elif self.model.decoder == 'transe':
                    dist = (h[edges[:, :, 0]] + self.model.rela_embed.weight[edges[:, :, 1]]
                            - h[edges[:, :, 2]]).norm(p=2, dim=-1)
                    score = 1 - nn.functional.sigmoid(dist)
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

    def get_config(self):
        return
