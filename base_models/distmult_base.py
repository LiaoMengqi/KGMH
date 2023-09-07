import torch
import torch.nn as nn
import math


class DistMultBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 input_dim,
                 output_dim):
        super(DistMultBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = DistMultEncoder(num_entity, input_dim, output_dim)
        self.decoder = DistMultDecoder(num_relation, output_dim)

    def forward(self, sub, rela, obj):
        sub_embed = self.encoder(sub)
        obj_embed = self.encoder(obj)
        return self.decoder(sub_embed, obj_embed, rela)


class DistMultEncoder(nn.Module):
    def __init__(self,
                 num_entity,
                 input_dim,
                 output_dim):
        super(DistMultEncoder, self).__init__()
        self.entity_embed = torch.nn.Embedding(num_entity, input_dim)
        self.linear = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim),
                                          torch.nn.ReLU())
        self.init_weight()

    def init_weight(self):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.linear[0].weight, gain=gain)

    def forward(self, index):
        return self.linear(self.entity_embed(index))


class DistMultDecoder(nn.Module):
    def __init__(self,
                 num_relation,
                 dim):
        super(DistMultDecoder, self).__init__()
        self.diag = nn.Parameter(torch.Tensor(num_relation, dim))
        self.init_weight()

    def init_weight(self):
        gain = nn.init.calculate_gain('sigmoid')
        std = gain * math.sqrt(2 / (1 + self.diag.shape[1]))
        nn.init.normal_(self.diag, mean=0, std=std)

    def forward(self,
                sub_embed,
                obj_embed,
                rela):
        return torch.sum(sub_embed * self.diag[rela] * obj_embed, dim=-1)
