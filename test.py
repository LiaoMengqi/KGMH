import torch.optim
from utils.plot import hist_value
from data.data_loader import DataLoader
from base_models.cen_base import CENBase
from models.cen import CEN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
epochs = 10
batch_size = 128
step = 1

data = DataLoader('ICEWS14s', './data/temporal/extrapolation')
data.load(load_time=True)
data.to(device)

base = CENBase(data.num_entity, data.num_relation, dim=64, dropout=0.2)
base.to(device)
opt = torch.optim.Adam(base.parameters(), lr=1e-3, weight_decay=1e-5)
model = CEN(base, data, opt)
model.to(device)

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
