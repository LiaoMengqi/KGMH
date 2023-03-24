import torch.optim
from data_loader import DataLoader
from models.transe import TransE
from base_models.transe_base import TransEBase

# general hyper parameters
epochs = 1000
batch_size = 500
lr = 0.00001
# model special hyper parameters
dim = 50
margin = 1
c = 1
# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data
data = DataLoader('FB15k-237', './data/static')
data.load()
data.to(device)
# base model
transe_base = TransEBase(data.num_entity, data.num_relation, emb_dim=dim, margin=margin, c=c).to(device)
# optimizer
opt = torch.optim.Adam(transe_base.parameters(), lr=lr)
# model
transe = TransE(transe_base, data, opt)

for epoch in range(epochs):
    out = ''
    loss = transe.train_epoch(batch_size)
    out = out + 'loss: ' + str(loss) + '| '
    if (epoch + 1) % 5 == 0:
        metric_dict = transe.test(batch_size, metric_list=['hist@10', 'mr'])
        for key in metric_dict.keys():
            out = out + key + ': ' + str(metric_dict[key]) + '| '
    print(out)
