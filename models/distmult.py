from base_models.distmult_base import DistMultBase
from models.mate_model import MateModel
from data.data_loader import DataLoader
import utils.data_process as dps
import utils.metrics as mtc

from tqdm import tqdm
import torch


class DistMult(MateModel):
    def __init__(self, model: DistMultBase, data: DataLoader, opt: torch.optim.Optimizer):
        super(DistMult, self).__init__()
        super().__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.name = 'distmult'

        # for filter
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
            pos_score = self.model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = self.model(neg_sample[:, 0], neg_sample[:, 1], neg_sample[:, 2])
            loss = self.loss(pos_score, neg_score)
            loss.backward()
            self.opt.step()
            total_loss += float(loss)
        return total_loss / total_batch

    def loss(self,
             pos_score,
             neg_score):
        return torch.nn.functional.relu(1 - pos_score + neg_score).mean()

    def predict(self, query):
        y_h = self.model.encoder.linear(self.model.encoder.entity_embed(query[:, 0]))
        y_t = self.model.encoder.linear(self.model.encoder.entity_embed.weight)
        score = torch.sum((y_h * self.model.decoder.diag[query[:, 1]]).unsqueeze(1) * y_t.unsqueeze(0), dim=-1)
        return score

    def test(self, batch_size=128, dataset='valid', metric_list=None, filter_out=False):
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
        with torch.no_grad():
            for batch_index in tqdm(range(total_batch)):
                batch = next(data)
                with torch.no_grad():
                    score = self.predict(batch[:, :2])
                    rank = mtc.calculate_rank(score, batch[:, 2])
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

    def get_config(self):
        config = {}
        config['model'] = self.name
        config['dataset'] = self.data.dataset
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['input_dim'] = self.model.input_dim
        config['output_dim'] = self.model.output_dim
        return config
