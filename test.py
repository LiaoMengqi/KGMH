import torch.optim

from data.data_loader import DataLoader
from base_models.cygnet_base import CyGNetBase
from models.cygnet import CyGNet
from utils.plot import hist_value

# general hyper parameters
epochs = 30
lr = 0.001
batch_size = 1024
# model special hyper parameters
dim = 200
alpha = 0.7
penalty = -100
reg_fact = 0.01
# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data
data = DataLoader('ICEWS14s', './data/temporal/extrapolation')
data.load(load_time=True)
data.to(device)
# base model
model = CyGNetBase(data.num_entity, data.num_relation, dim, alpha=alpha)
model.to(device)
# optimizer
opt = torch.optim.Adam(params=model.parameters(), lr=lr)
# model
rgcn = CyGNet(model, data, opt)
rgcn.to(device)

loss_list = []
history_metric_dict = {}
step = 5
for epoch in range(epochs):
    if (epoch + 1) % step == 0:
        loss = rgcn.train_epoch(batch_size)
        torch.cuda.empty_cache()
        metrics = rgcn.test(batch_size=batch_size)
        torch.cuda.empty_cache()
        print('epoch:%d |loss: %f |hist@10: %f |hist@100: %f |mr: %f' % (
            epoch + 1, loss, metrics['hist@10'], metrics['hist@100'], metrics['mr']))
        for key in metrics.keys():
            if key not in history_metric_dict.keys():
                history_metric_dict[key] = []
            history_metric_dict[key].append(metrics[key])
    else:
        loss = rgcn.train_epoch(batch_size=batch_size)
        torch.cuda.empty_cache()
        print('epoch:%d |loss: %f' % (epoch + 1, loss))
    loss_list.append(loss)

hist_value({'loss': loss_list}, name='Loss')
hist_value({'hist@10': history_metric_dict['hist@10'], 'hist@100': history_metric_dict['hist@100']}, name='hist@k')
hist_value({'mr': history_metric_dict['mr']}, name='mr')
