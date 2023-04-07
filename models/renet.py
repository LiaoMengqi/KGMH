import torch
import torch.nn as nn
from base_models.renet_base import ReNetGlobalBase
from data.data_loader import DataLoader
import utils.data_process as dps
import torch.nn.functional as F


class ReNetGlobal(nn.Module):
    def __init__(self,
                 model: ReNetGlobalBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(ReNetGlobal, self).__init__()
        self.model = model
        self.opt = opt
        self.data = data
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.true_prob_train = self.calculate_prob(self.train_data, data.num_entity)
        self.true_prob_valid = self.calculate_prob(self.valid_data, data.num_entity)
        self.true_prob_test = self.calculate_prob(self.test_data, data.num_entity)

    @staticmethod
    def calculate_prob(edges: list,
                       num_entity):
        prob_list = []
        for edge in edges:
            src = edge[:, 0]
            src_unique, count = torch.unique(src, return_counts=True)
            prob = torch.zeros(num_entity, device=src.device)
            prob[src_unique] = count / num_entity
            prob_list.append(prob.unsqueeze(0))
        return torch.cat(prob_list, dim=0)

    def soft_cross_entropy(self,
                           pred,
                           soft_targets):
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), dim=1))

    def train_epoch(self,
                    batch_size: int):
        self.train()
        self.opt.zero_grad()
        score, index = self.model.forward(self.train_data)
        loss = self.soft_cross_entropy(score, self.true_prob_train[index])
        loss.backward()
        self.opt.step()
        return float(loss)

    def test(self, edges):
        """
        Predict s at time t,  graphs[t-seq_len:t-1] are used.
        :return:
        """

        return
