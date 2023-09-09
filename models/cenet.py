import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from base_models.cenet_base import CeNetBase
from data.data_loader import DataLoader
import utils.data_process as dps
import utils.metrics as mtc
import utils
from models.mate_model import MateModel


class CeNet(MateModel):
    def __init__(self,
                 model: CeNetBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(CeNet, self).__init__()
        self.model = model
        self.opt = opt
        self.data = data
        self.name = 'cenet'
        # data process
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.train_data = dps.add_reverse_relation(self.train_data, self.data.num_relation)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.valid_data = dps.add_reverse_relation(self.valid_data, self.data.num_relation)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.test_data = dps.add_reverse_relation(self.test_data, self.data.num_relation)
        # record history
        self.history = torch.sparse_coo_tensor(
            size=(
                len(self.train_data) + len(self.valid_data) + len(self.test_data),
                data.num_entity * data.num_relation * 2,
                data.num_entity),
            device=data.train.device)
        self.update_history(self.train_data + self.valid_data + self.test_data)

        self.grad_norm = 1.0
        self.cross_entropy = nn.BCELoss()

    def update_history(self, dataset):
        for index, data in enumerate(dataset):
            i0 = torch.zeros(data.shape[0], device=self.data.device, dtype=torch.long) + index
            i1 = data[:, 0] * self.data.num_relation * 2 + data[:, 1]
            i = torch.cat([i0.unsqueeze(0), i1.unsqueeze(0), data[:, 2].unsqueeze(0)], dim=0)
            v = torch.ones(i.shape[-1])
            snap = torch.sparse_coo_tensor(i, v, size=self.history.shape, device=self.data.device)
            self.history.add_(snap)
            if index > 0:
                self.history[index].add_(self.history[index - 1])

    def train_epoch(self,
                    batch_size=128):
        self.model.train()
        total_loss = 0
        for i in tqdm(range(len(self.train_data))):
            if i == 0:
                history = torch.sparse_coo_tensor(size=self.history[0].shape, device=self.data.device)
            else:
                history = self.history[i - 1]
            for batch in utils.data_process.batch_data(self.train_data[i], batch_size=batch_size):
                b_history = torch.index_select(history, index=batch[:, 0] * self.data.num_relation * 2 + batch[:, 1],
                                               dim=0).to_dense()
                h_his, h_nhis = self.model.forward(batch, b_history)
                loss_main = self.his_loss(h_his, h_nhis, batch[:, 2])
                loss_cts = self.contrastive_loss(batch[:, 0], batch[:, 1], batch[:, 2], b_history)
                loss = loss_main * self.model.alpha + loss_cts * (1 - self.model.alpha)
                self.opt.zero_grad()
                loss.backward()
                total_loss = total_loss + loss.item()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                self.opt.step()
        for i in tqdm(range(len(self.train_data))):
            if i == 0:
                history = torch.sparse_coo_tensor(size=self.history[0].shape, device=self.data.device)
            else:
                history = self.history[i - 1]
            for batch in utils.data_process.batch_data(self.train_data[i], batch_size=batch_size):
                b_history = torch.index_select(history, index=batch[:, 0] * self.data.num_relation * 2 + batch[:, 1],
                                               dim=0).to_dense()
                classify_loss = self.classifier_loss(batch[:, 0], batch[:, 1], batch[:, 2], b_history)
                self.opt.zero_grad()
                classify_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
                self.opt.step()
        return total_loss

    def contrastive_loss(self, sub, rela, obj, freq):
        label = freq[torch.arange(obj.shape[0]), obj]
        label[label > 0] = 1
        v_query = self.model.linear_query(
            torch.cat([self.model.entity_embeds[sub],
                       self.model.rel_embeds[rela],
                       F.tanh(self.model.linear_freq(freq))],
                      dim=1)
        )
        dot_product_matrix = v_query.mm(v_query.T)
        exp_matrix = torch.exp(dot_product_matrix - torch.max(dot_product_matrix, dim=1, keepdim=True)[0]) + 1e-5
        same_matrix = label.unsqueeze(1).repeat(1, label.shape[0]) == label
        mask = 1 - torch.eye(exp_matrix.shape[0], device=self.data.device)
        same_masked = same_matrix * mask
        cardinality = torch.sum(same_masked, dim=1) + 1e-3
        log_res = -torch.log(exp_matrix / (torch.sum(exp_matrix * mask, dim=1, keepdim=True)))
        l_sup = torch.sum(log_res * same_masked, dim=1) / cardinality
        l_sup = torch.nan_to_num(l_sup)
        return torch.mean(l_sup)

    def his_loss(self,
                 h_his,
                 h_nhis,
                 obj):
        l_ce = -torch.sum(
            torch.gather(torch.log_softmax(h_his, dim=1) + torch.log_softmax(h_nhis, dim=1), 1, obj.view(-1, 1)))
        return l_ce / h_his.shape[0]

    def classify(self, sub, rela, history):

        self.model.freeze_embed(False)
        freq_h = self.model.linear_freq(history)
        pred = F.sigmoid(self.model.bi_classifier(torch.cat([self.model.entity_embeds[sub],
                                                             self.model.rel_embeds[rela],
                                                             freq_h], dim=1)))
        self.model.freeze_embed(True)
        return pred

    def classifier_loss(self, sub, rela, obj, history):
        label = history[torch.arange(obj.shape[0]), obj]
        label[label > 0] = 1
        pred = self.classify(sub, rela, history)
        return self.cross_entropy(pred.squeeze(), label.squeeze())

    def test(self,
             batch_size=128,
             dataset='valid',
             metric_list=None,
             filter_out=False):
        self.eval()
        if metric_list is None:
            metric_list = ['hits@1', 'hits@3', 'hits@10', 'hits@100', 'mr', 'mrr']
        if dataset == 'valid':
            data = self.valid_data
            history_start = len(self.train_data)
        elif dataset == 'test':
            data = self.test_data
            history_start = len(self.train_data) + len(self.valid_data)
        else:
            raise Exception
        rank_list = []
        rank_list_filter = []
        with torch.no_grad():
            for i in tqdm(range(len(data))):
                history = self.history[history_start + i - 1]
                for batch in utils.data_process.batch_data(data[i]):
                    b_history = torch.index_select(history,
                                                   index=batch[:, 0] * self.data.num_relation * 2 + batch[:, 1],
                                                   dim=0).to_dense()
                    h_his, h_nhis = self.model.forward(batch, b_history)
                    pred_p = (torch.softmax(h_nhis, dim=1) + torch.softmax(h_his, dim=1)) / 2
                    pred_c = self.classify(batch[:, 0], batch[:, 1], b_history)
                    pred_c[pred_c >= 0.5] = 1.0
                    pred_c[pred_c < 0.5] = 0
                    pred_c = pred_c.expand(pred_c.shape[0], self.data.num_entity)
                    entity_with_his = b_history.clone()
                    entity_with_his[entity_with_his > 0] = 1.0
                    mask = (entity_with_his == pred_c).float()
                    if self.model.mode == 'soft':
                        mask = torch.softmax(mask, dim=1)
                    score = pred_p * mask
                    ranks = mtc.calculate_rank(score, batch[:, 2])
                    rank_list.append(ranks)
                    if filter_out:
                        ans = utils.data_process.get_answer(data[i], self.data.num_entity, self.data.num_relation * 2)
                        score = utils.data_process.filter_score(score, ans, batch, self.data.num_relation * 2)
                        rank = mtc.calculate_rank(score, batch[:, 2])
                        rank_list_filter.append(rank)
        all_ranks = torch.cat(rank_list)
        metrics = mtc.ranks_to_metrics(metric_list=metric_list, ranks=all_ranks)
        if filter_out:
            all_rank = torch.cat(rank_list_filter)
            metrics_filter = mtc.ranks_to_metrics(metric_list, all_rank, filter_out)
            metrics.update(metrics_filter)
        return metrics

    def get_config(self):
        config = {}
        config['model'] = 'cenet'
        config['dataset'] = self.data.dataset
        config['num_entity'] = self.model.num_entity
        config['num_relation'] = self.model.num_relation
        config['dim'] = self.model.dim
        config['drop_prop'] = self.model.drop_prop
        config['lambdax'] = self.model.lambdax
        config['alpha'] = self.model.alpha
        config['mode'] = self.model.mode
        return config
