import torch
import torch.nn as nn
from tqdm import tqdm

from base_models.cen_base import CENBase
from data.data_loader import DataLoader
import utils.data_process as dps


class CEN(nn.Module):
    def __init__(self,
                 model: CENBase,
                 data: DataLoader,
                 opt: torch.optim.Optimizer):
        super(CEN, self).__init__()
        self.model = model
        self.data = data
        self.opt = opt
        self.train_data, _, self.train_time = dps.split_data_by_time(self.data.train)
        self.valid_data, _, self.valid_time = dps.split_data_by_time(self.data.valid)
        self.test_data, _, self.test_time = dps.split_data_by_time(self.data.test)
        self.k = model.k
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def train_epoch(self,
                    batch_size=128):
        self.model.train()
        self.opt.zero_grad()
        # add reverse relation to graph
        data = dps.add_reverse_relation(self.train_data, self.data.num_relation)
        # target time for predict
        total_loss = 0
        for i in tqdm(range(len(data))):
            if i == 0:
                # no history data
                continue
            # history data
            if i >= self.k:
                history_graphs = data[i - self.k:i]
            else:
                history_graphs = data[0:i]
            score = self.model.forward(history_graphs,
                                       data[i])
            loss = self.loss(score, data[i][:, 2])
            loss.backward()
            total_loss = total_loss + float(loss)
            self.opt.step()
        return total_loss

    def loss(self,
             score,
             target):
        return self.cross_entropy_loss(score, target)

    def test(self,
             batch_size=128):
        return
