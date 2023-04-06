import torch.optim

from data.data_loader import DataLoader
from base_models.regcn_base import RGCNBase
from models.rgcn import RGCN
from utils.plot import hist_value

# general hyper parameters
epochs = 50
lr = 0.01
weight_decay = 0

# model special hyper parameters
dims = [150, 150]
num_block = 30
b = 5
w = 1

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data
data = DataLoader('FB15k-237', './data/static')
data.load()
data.to(device)

# base model
model = RGCNBase(dims, data.num_relation,
                 data.num_entity,
                 use_block=True,
                 num_block=num_block,
                 dropout_s=0.2,
                 dropout_o=0.4)
model.to(device)
# optimizer
opt = torch.optim.Adam(params=model.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)
# model
rgcn = RGCN(model,
            data,
            opt,
            w=w)

rgcn.to(device)

loss_list = []
step = 5
history = {}
for epoch in range(epochs):
    if (epoch + 1) % step == 0:
        loss = rgcn.train_epoch(batch_size=100000, save_h=True)
        torch.cuda.empty_cache()
        metrics = rgcn.test(batch_size=75, metric_list=['hist@100', 'mr', 'hist@10'])
        for key in metrics.keys():
            if key in history.keys():
                history[key].append(metrics[key])
            else:
                history[key] = []
                history[key].append(metrics[key])
        print('epoch:%d |loss: %f |hist@100: %f |hist@10: %f |mr: %f' % (
            epoch + 1, loss, metrics['hist@100'], metrics['hist@10'], metrics['mr']))
    else:
        loss = rgcn.train_epoch(batch_size=100000)
        print('epoch:%d |loss: %f' % (epoch + 1, loss))
    loss_list.append(loss)

hist_value({'loss': loss_list}, name='loss')
hist_value({'hist@10': history['hist@10'], 'hist@100': history['hist@100']}, name='hist@k')
hist_value({'mr': history['mr']}, name='mr')
