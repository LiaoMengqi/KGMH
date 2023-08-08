import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models.regcn_base import URGCNBase


class CENBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 dim,
                 dropout=0.0,
                 channel=50,
                 width=3,
                 seq_len=10,
                 layer_norm=True):
        super(CENBase, self).__init__()
        self.num_relation = num_relation
        self.channel = channel
        self.width = width
        self.seq_len = seq_len
        self.dim = dim
        self.num_entity = num_entity
        self.layer_norm = layer_norm
        self.dropout_value = dropout

        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, dim))
        self.relation_embed = nn.Parameter(torch.Tensor(num_relation * 2, dim))

        self.encoder = KGSEncoder(dim,
                                  num_relation * 2,
                                  dropout=dropout)
        self.decoder = ERDecoder(dim,
                                 seq_len=seq_len,
                                 num_channel=channel,
                                 kernel_length=width,
                                 dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channel * dim, dim, bias=False)
        self.bn_list = torch.nn.ModuleList()
        for _ in range(seq_len):
            self.bn_list.append(torch.nn.BatchNorm1d(dim))
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.entity_embed)
        torch.nn.init.xavier_normal_(self.relation_embed)

    def forward(self,
                history_graph: list,
                target_graph,
                training=True):
        length = len(history_graph)
        score = torch.zeros(len(target_graph), self.num_entity, device=target_graph.device)
        for i in range(self.seq_len):
            # length from 1 to k
            if i + 1 > length:
                break
            x = self.encoder(history_graph[-(i + 1):],
                             entity_embed=self.entity_embed,
                             relation_embed=self.relation_embed,
                             training=training)
            # size=(num_query,dim*c)
            x = self.decoder(x,
                             self.relation_embed,
                             target_graph[:, [0, 1]],
                             i,
                             training=training)
            # size=(num_query,dim)
            x = self.fc(x)
            x = self.dropout(x)
            x = self.bn_list[i](x)
            F.relu_(x)
            # size=(num_query,num_entity)
            x = torch.mm(x, self.entity_embed.transpose(0, 1))
            score.add_(x)
        return score


class KGSEncoder(nn.Module):
    def __init__(self,
                 dim,
                 num_relation,
                 dropout=0.0,
                 layer_norm=True):
        super(KGSEncoder, self).__init__()
        self.layer_norm = layer_norm
        self.rgcn = URGCNBase(dim,
                              num_layer=2,
                              active=False,
                              dropout=dropout,
                              self_loop=True)
        self.time_gate_weight = nn.Parameter(torch.Tensor(dim, dim))
        self.time_gate_bias = nn.Parameter(torch.Tensor(dim))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.time_gate_bias)

    def normalize(self, tensor):
        """
        perform F2 Normalization
        :param tensor: tensor to be normalized
        :return: return tensor normalized if using layer normalization ,else return origin tensor
        """
        return F.normalize(tensor) if self.layer_norm else tensor

    def forward(self,
                edges: list,
                entity_embed,
                relation_embed,
                training=True):
        entity_embed = self.normalize(entity_embed)
        relation_embed = self.normalize(relation_embed)
        for edge in edges:
            current_embed = self.rgcn.forward(entity_embed, relation_embed, edge, training)
            current_embed = self.normalize(current_embed)
            time_weight = F.sigmoid(torch.mm(entity_embed, self.time_gate_weight) + self.time_gate_bias)
            entity_embed = time_weight * current_embed + (1 - time_weight) * entity_embed
            entity_embed = self.normalize(entity_embed)
        return entity_embed


class ERDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 seq_len,
                 num_channel=50,
                 kernel_length=3,
                 dropout=0.0,
                 bias=False):
        super(ERDecoder, self).__init__()
        self.input_dim = input_dim
        self.c = num_channel
        self.k = kernel_length
        self.dropout = torch.nn.Dropout(dropout)
        self.bn0_list = torch.nn.ModuleList()
        self.bn1_list = torch.nn.ModuleList()

        self.conv_list = torch.nn.ModuleList()
        if kernel_length % 2 != 0:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2), int(kernel_length / 2), 0, 0))
        else:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2) - 1, int(kernel_length / 2), 0, 0))

        for _ in range(seq_len):
            self.conv_list.append(nn.Conv2d(1, num_channel, (2, kernel_length)))
            self.bn0_list.append(torch.nn.BatchNorm1d(2))
            self.bn1_list.append(torch.nn.BatchNorm1d(num_channel))

        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_dim * num_channel, input_dim, bias=bias)

    def forward(self,
                entity_ebd,
                relation_ebd,
                query,
                seq_len,
                training=True):
        x = torch.cat([entity_ebd.unsqueeze(dim=1)[query[:, 0]],
                       relation_ebd.unsqueeze(dim=1)[query[:, 1]]],
                      dim=1)
        x = self.bn0_list[seq_len](x)
        if training:
            x = self.dropout(x)
        x.unsqueeze_(dim=1)
        x = self.conv_list[seq_len](self.pad(x))
        x = self.bn1_list[seq_len](x.squeeze())
        x = F.relu(x)
        if training:
            x = self.dropout(x)
        x = self.flat(x)
        return x
