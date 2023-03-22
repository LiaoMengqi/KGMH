import torch.nn as nn

import utils.data_process as dpc
from data_loader import DataLoader


class BaseModel(nn.Module):
    def __init__(self, data: DataLoader):
        super(BaseModel, self).__init__()
        self.data = data

    def batch(self, batch_size, data_set='train'):
        if data_set == 'train':
            self.data_batched = dpc.batch_data(self.data.train, batch_size)
        elif data_set == 'valid':
            self.data_batched = dpc.batch_data(self.data.valid, batch_size)
        elif data_set == 'test':
            self.data_batched = dpc.batch_data(self.data.test, batch_size)
        raise Exception

    def train_step(self, batch_size, opt):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
