import torch.optim

from data.data_loader import DataLoader
from base_models.regcn_base import REGCNBase
from models.regcn import REGCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 32
hidden_dim = 32
num_k = 4
seq_len = 10
num_layer = 2

dropout = 0.2
weight_decay = 1e-5
lr = 0.001
epochs = 10
batch_size = 64

data = DataLoader('ICEWS14s', './data/temporal/extrapolation')
data.load(load_time=True)
data.to(device)
base = REGCNBase(data.num_entity,
                 data.num_relation,
                 hidden_dim,
                 seq_len,
                 num_layer,
                 dropout=dropout,
                 active=True
                 )
base.to(device)

opt = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=weight_decay)
model = REGCN(base, data, opt)
model.to(device)

step = 1
for i in range(epochs):
    loss = model.train_epoch()
    print('loss: %f' % loss)
    if i % step == 0:
        metrics = model.test()
        for key in metrics.keys():
            print(key + ' : ' + str(metrics[key]))
