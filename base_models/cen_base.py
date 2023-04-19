import torch
import torch.nn as nn
from base_models.rgcn_base import RGCNBase
import torch.nn.functional as F


class CENBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 dim,
                 dropout=0.0,
                 c=50,
                 w=3,
                 k=10):
        super(CENBase, self).__init__()
        self.c = c
        self.w = w
        self.k = k
        self.dim = dim
        self.num_entity = num_entity

        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, dim))
        self.relation_embed = nn.Parameter(torch.Tensor(num_relation, dim))

        self.encoder = KGSEncoder(dim,
                                  num_relation)
        self.decoder = ERDecoder(dim,
                                 num_channel=c,
                                 kernel_length=w,
                                 dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * dim, dim, bias=False)
        self.bn = torch.nn.BatchNorm1d(dim)

    def init_weight(self):
        pass

    def forward(self,
                history_graph: list,
                target_graph):
        length = len(history_graph)
        score = torch.zeros(len(target_graph), self.num_entity)
        for i in range(self.k):
            # length from 1 to k
            if i + 1 > length:
                break
            entity_evolved_embed = self.encoder(history_graph[-(i + 1):], h_input=self.entity_embed)
            # size=(num_query,dim*c)
            x = self.decoder(entity_evolved_embed, self.relation_embed, target_graph[:, [0, 1]])
            # size=(num_query,dim)
            x = self.fc(x)
            x = self.dropout(x)
            x = self.bn(x)
            x = F.relu(x)
            # size=(num_query,num_entity)
            x = torch.mm(x, self.entity_embed.transpose(0, 1))
            score = score + x
        return score


class KGSEncoder(nn.Module):
    def __init__(self,
                 dim,
                 num_relation):
        super(KGSEncoder, self).__init__()
        self.rgcn = RGCNBase([dim, dim, dim], num_relation)

    def forward(self,
                edges: list,
                entity_embed_init):
        entity_embed = entity_embed_init
        for edge in edges:
            entity_embed = self.rgcn.forward(edge, h_input=entity_embed)
        return entity_embed


class ERDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channel,
                 kernel_length,
                 dropout=0.0,
                 bias=False):
        super(ERDecoder, self).__init__()
        self.input_dim = input_dim
        self.c = num_channel
        self.k = kernel_length
        self.dropout = torch.nn.Dropout(dropout)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.c)

        if kernel_length % 2 != 0:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2), int(kernel_length / 2), 0, 0))
        else:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2) - 1, int(kernel_length / 2), 0, 0))
        self.conv = nn.Conv2d(1, num_channel, (2, kernel_length))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_dim * num_channel, input_dim, bias=bias)

    def forward(self, entity_ebd, relation_ebd, query):
        x = torch.cat([entity_ebd.unsqueeze(dim=1)[query[:, 0]],
                       relation_ebd.unsqueeze(dim=1)[query[:, 1]]],
                      dim=1)
        x = self.bn0(x)
        x = self.dropout(x)
        x.unsqueeze_(dim=1)
        x = self.conv(self.pad(x))
        x = self.bn1(x.squeeze())
        x = F.relu(x)
        x = self.dropout(x)
        x = self.flat(x)
        return x
