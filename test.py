import numpy as np
import torch.optim
from tqdm import tqdm

import utils.metrics as mt
from data_loader import DataLoader
from modules.encoder import TransE
from utils import data_process as dp

# general
epochs = 1000
batch_size = 500
lr = 0.01

# model special
dim = 50
margin = 1
c = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = DataLoader('FB15k-237', './data/static')
data.load()
valid_batched = dp.batch_data(data.valid, batch_size)
transe = TransE(data.num_entity, data.num_relation, emb_dim=dim, margin=margin, c=c).to(device)
opt = torch.optim.Adam(transe.parameters(), lr=lr)

data.to(device)
for epoch in range(epochs):
    transe.train()
    train_batched = dp.batch_data(data.train, batch_size)
    loss_all = 0.0
    num_sample = 0
    for batch in tqdm(train_batched):
        transe.zero_grad()
        neg_sample = dp.generate_negative_sample(batch, data.num_entity)
        loss = transe.loss(batch, neg_sample)
        loss.backward()
        loss_all = loss_all + float(loss)
        num_sample = num_sample + batch.shape[0]
        opt.step()
    ave_loss = loss_all / num_sample

    transe.eval()
    rank_list = []
    if (epoch + 1) % 1 == 0:
        for batch in tqdm(valid_batched):
            dis = transe.predict(batch[:, 0], batch[:, 1])
            ans = batch[:, 1].cpu().numpy()
            dis = -dis.detach().cpu().numpy()
            rank = mt.calculate_rank(dis, ans)
            rank_list.append(rank)
        all_rank = np.concatenate(rank_list)
        hist100 = mt.calculate_hist(100, all_rank)
        hist3 = mt.calculate_hist(3, all_rank)
        hist10 = mt.calculate_hist(10, all_rank)
        print('epoch: ' + str(epoch + 1) + ' | loss: ' + str(ave_loss) + ' | hist@100: ' + str(
            hist100) + ' | hist@3: ' + str(
            hist3) + ' | hist@10: ' + str(hist10))
    else:
        print('epoch: ' + str(epoch + 1) + ' | loss: ' + str(ave_loss))
