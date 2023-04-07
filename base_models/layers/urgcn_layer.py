import torch
import torch.nn as nn
from base_models.layers.rgcn_layer import RGCNLayer


class URGCNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 active=False,
                 dropout=0.0,
                 self_loop=True,
                 skip_connect=False):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param active:
        """
        super(URGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # components

        self.w_neighbor = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.dropout = nn.Dropout(dropout)

        if self_loop:
            self.w_self = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.w_self_evolve = nn.Parameter(torch.Tensor(input_dim, output_dim))
        else:
            self.self_loop = None

        if active:
            self.active = nn.ReLU()
        else:
            self.active = None

    def init_weight(self):
        nn.init.xavier_uniform_(self.w_neighbor, gain=nn.init.calculate_gain('relu'))
        if self.self_loop is not None:
            nn.init.xavier_uniform_(self.w_self, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.w_self_evolve, gain=nn.init.calculate_gain('relu'))

    def forward(self, nodes_embed, relation_embed, edges):
        """
        :param nodes_embed:Tensor, size=(num_node,input_dim)
        :param relation_embed: Tensor,size=(num_edge,input_dim)
        :param edges: Tensor, size=(num_edge, 3), with the format of (source node, edge, destination node)
        :return: the representation of node after aggregation
        """
        # self loop
        if self.self_loop is not None:
            h = torch.mm(nodes_embed, self.w_self)
        else:
            h = nodes_embed
        # calculate message
        message = torch.mm(nodes_embed[edges[:, 0]] + relation_embed[edges[:, 1]],
                           self.w_neighbor)
        # aggregate
        des_index, message = RGCNLayer.aggregate(message, edges[:, 2])
        # send message
        h[des_index] = h[des_index] + message
        if self.active is not None:
            h = self.active(h)
        h = self.dropout(h)
        return h
