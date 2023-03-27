import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, num_relation, dim):
        super(DistMult, self).__init__()
        self.diag = nn.Parameter(torch.Tensor(num_relation, dim))
        self.weight_init()

    def weight_init(self):
        nn.init.normal_(self.diag)
        pass

    def forward(self, sub_embed, obj_embed, rela):
        return torch.sum(sub_embed * self.diag[rela] * obj_embed, dim=-1)
