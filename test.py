import torch

from data.data_loader import DataLoader
from base_models.distmult_base import DistMultBase
import utils.data_process as dps

data = DataLoader(dataset='FB15k-237', root_path='./data/', type='static')
data.load()

model = DistMultBase(data.num_entity, data.num_relation, 32, 32)

sub = torch.LongTensor([1, 2, 3])
rela = torch.LongTensor([5, 1, 2])
obj = torch.LongTensor([0, 9, 14])

res = model.forward(sub, rela, obj)
res.sum().backward()
print(res)
