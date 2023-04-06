import torch
import torch.nn as nn
import math


class DistMult(nn.Module):
    def __init__(self,
                 num_relation,
                 dim):
        super(DistMult, self).__init__()
        self.diag = nn.Parameter(torch.Tensor(num_relation, dim))
        self.weight_init()

    def weight_init(self):
        gain = nn.init.calculate_gain('sigmoid')
        std = gain * math.sqrt(2 / (1 + self.diag.shape[1]))
        nn.init.normal_(self.diag, mean=0, std=std)

    def forward(self,
                sub_embed,
                obj_embed,
                rela):
        return torch.sum(sub_embed * self.diag[rela] * obj_embed, dim=-1)
