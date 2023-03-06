from modules.decoder import Conv_TransE
import torch

model = Conv_TransE(16, 3, 3)

node_ebd = torch.ones((30, 16), dtype=torch.float64)
edge_ebd = torch.ones((10, 16), dtype=torch.float64)
src = torch.randint(0, 30, (1, 20))
des = torch.randint(0, 30, (1, 20))
rela = torch.randint(0, 10, (1, 20))
edge = torch.cat([src, rela, des], dim=0).T
res = model.forward(node_ebd, edge_ebd, edge[:, [0, 1]])
