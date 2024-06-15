import torch
from data.data_loader import DataLoader
from models.ggecddcd import GGEcdDcd
from base_models.ggecddcd_base import GGEcdDcdBase
from utils.func import save_json
from utils.func import set_seed
import time

device = 'cuda:2'
set_seed(0)

# hyper param
encoder = 'gcn'
decoder = 'distmult'
input_dim = 50
output_dim = 50
dataset = 'FB15k-237'
hidden_dims = [50]
# fb15k 1e-3 1e-3 1e-2
lr = 1e-3

data = DataLoader(dataset=dataset, root_path='./data/', type='static')
data.load()
data.to(device)

base_models = GGEcdDcdBase(encoder=encoder,
                           decoder=decoder,
                           num_entity=data.num_entity,
                           num_relation=data.num_relation,
                           input_dim=input_dim,
                           output_dim=output_dim,
                           hidden_dims=hidden_dims)
base_models.to(device)
x = base_models.parameters()
opt = torch.optim.Adam(base_models.parameters(), lr=lr)

model = GGEcdDcd(base_models, data, opt)
epochs = 100

history = []
stop_step = 3
tolerance = 3
best = 0
model_id = time.strftime('%Y%m%d%H%M%S', time.localtime())

for epoch in range(epochs):
    model.train_epoch(batch_size=3072)
    print('epoch ', epoch)
    metrics = model.test(batch_size=256, filter_out=True)
    print(metrics)
    history.append(metrics)
    if metrics['filter mrr'] > history[best]['filter mrr']:
        best = epoch
        tolerance = stop_step
    else:
        tolerance -= 1
    if tolerance < 1:
        break

print(history[best])

info = {'encoder': encoder,
        'decoder': decoder,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'dataset': dataset,
        'hidden_dims': hidden_dims,
        'lr': lr}

save_json(history, path='./checkpoint/ggecddcd/' + model_id + '/', name='history')
save_json(info, path='./checkpoint/ggecddcd/' + model_id + '/', name='info')
save_json(history[best], path='./checkpoint/ggecddcd/' + model_id + '/', name='best')
