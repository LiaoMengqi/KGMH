import torch.optim
from utils.plot import hist_value
from data.data_loader import DataLoader
from base_models.regcn_base import REGCNBase
from models.regcn import REGCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 32
hidden_dim = 32
seq_len = 10
num_layer = 2

dropout = 0.2
weight_decay = 1e-5
lr = 0.001
epochs = 20

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
metric_history = {}
loss_history = []
for i in range(epochs):
    loss = model.train_epoch()
    loss_history.append(loss)
    print('epoch :%d |loss: %f' % (i + 1, loss))
    if i % step == 0:
        metrics = model.test()
        print('hist@3: %f |hist@10: %f |mr: %f' % (metrics['hist@3'], metrics['hist@10'], metrics['mr']))
        for key in metrics.keys():
            if key in metric_history.keys():
                metric_history[key].append(metrics[key])
            else:
                metric_history[key] = []
                metric_history[key].append(metrics[key])

hist_value({'hist@10': metric_history['hist@10'],
            'hist@3': metric_history['hist@3']}, name='hist@k')
hist_value({'mr': metric_history['mr']}, name='mr')
hist_value({'loss': loss_history}, name='loss')
