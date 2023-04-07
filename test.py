import torch.optim

from data.data_loader import DataLoader
from models.renet import ReNetGlobal
from base_models.renet_base import ReNetGlobalBase

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 32
hidden_dim = 32
num_k = 4

lr = 0.01
epochs = 10
batch_size = 64

data = DataLoader('ICEWS14s', './data/temporal/extrapolation')
data.load(load_time=True)
data.to(device)
base = ReNetGlobalBase(data.num_entity,
                       data.num_relation,
                       input_dim,
                       hidden_dim,
                       dropout=0.2,
                       seq_len=10,
                       num_k=4).to(device)

opt = torch.optim.Adam(base.parameters(), lr=lr)

model = ReNetGlobal(base, data, opt).to(device)

for i in range(epochs):
    loss = model.train_epoch(batch_size)
    print('epoch %d| loss :%f |' % (i + 1, loss))
