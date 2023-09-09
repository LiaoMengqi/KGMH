from base_models.sacn_base import SACNBase

from data.data_loader import DataLoader

from models.mate_model import MateModel

import utils.data_process as dps
import utils.metrics as mtc

from tqdm import tqdm
import torch
import torch.nn as nn


class SACN(MateModel):
    def __init__(self,
                 model: SACNBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer
                 ):
        super(SACN, self).__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.name = 'sacn'

        self.cross_entropy = nn.CrossEntropyLoss()
        self.ans = None

    def train_epoch(self,
                    batch_size=128):
        self.train()
        self.opt.zero_grad()
        data = self.data.train
        total_loss = 0

        batches = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)

        for batch_index in tqdm(range(total_batch)):
            h_output = self.model.encoder.forward(self.model.entity_embed.weight, data)
            batch = next(batches)
            score = torch.sigmoid(self.model.decoder(h_output, self.model.relation_embed.weight, batch[:, :2]))
            # label smoothing
            label_smoothed = dps.label_smooth(batch[:, 2], self.data.num_entity, epsilon=0.1)
            loss = self.cross_entropy(score, label_smoothed)
            loss.backward()
            self.opt.step()
            total_loss += float(loss)

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

        data_batched = dps.batch_data(data, batch_size)
        total_batch = int(len(data) / batch_size) + (len(data) % batch_size != 0)

        rank_list = []
        rank_list_filter = []
        self.eval()
        with torch.no_grad():
            h = self.model.encoder(self.model.entity_embed.weight, self.data.train)
            for batch_index in tqdm(range(total_batch)):
                batch = next(data_batched)
                score = self.model.decoder(h, self.model.relation_embed.weight, batch[:, :2])
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
        config = {}
        config['model'] = self.name
        config['dataset'] = self.data.dataset

        config['num_relation'] = self.data.num_relation
        config['num_entity'] = self.data.num_entity

        config['dim'] = self.model.dim
        config['num_layer'] = self.model.num_layer
        config['num_channel'] = self.model.num_channel
        config['kernel_length'] = self.model.kernel_length
        config['dropout'] = self.model.drop_prop
        return config
