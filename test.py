import torch.optim

from data.data_loader import DataLoader
from base_models.regcn_base import RGCNBase
from models.rgcn import RGCN
from utils.plot import hist_value

# general hyper parameters
epochs = 10
lr = 0.001
# model special hyper parameters
dims = [36, 36]
b = 5
w = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = DataLoader('FB15k-237', './data/static')
data.load()
data.to(device)
model = RGCNBase(dims, data.num_relation, data.num_entity, basis=True, b=10)
model.to(device)
opt = torch.optim.Adam(params=model.parameters(), lr=lr)
rgcn = RGCN(model, data, opt, w=w)
rgcn.to(device)

loss_list = []
for epoch in range(epochs):
    if (epoch + 1) % 5 == 0:
        loss = rgcn.train_epoch(save_h=True)
        metrics = rgcn.test(200, metric_list=['hist@100', 'mr'])
        print('epoch:%d |loss: %f |hist@100: %f |mr: %f' % (epoch, loss, metrics['hist@100'], metrics['mr']))
    else:
        loss = rgcn.train_epoch(save_h=True)
        print('epoch:%d |loss: %f' % (epoch, loss))
    loss_list.append(loss)

hist_value({'loss': loss_list}, name='Loss')
