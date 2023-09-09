import torch

from base_models.sacn_base import SACNBase
from models.sacn import SACN
from data.data_loader import DataLoader

dim = 32

data = DataLoader(dataset='FB15k', root_path='./data/', type='static')
data.load()

base_model = SACNBase(data.num_entity, data.num_relation, dim, num_layer=2, num_channel=50,
                      kernel_length=3, dropout=0.2)
opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
model = SACN(base_model, data, opt)

loss = model.train_epoch(batch_size=2048)
print(loss)
met = model.test(batch_size=1024)
print(met)
