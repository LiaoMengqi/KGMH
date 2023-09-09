import torch
import torch.nn as nn
from base_models.layers.gnn import GNN
import torch.nn.functional as F


class URGCNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 active=False,
                 dropout=0.0,
                 self_loop=True):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param active:
        """
        super(URGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.active = active
        self.dropout = nn.Dropout(p=dropout)
        # components

        self.w_neighbor = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.self_loop = self_loop

        if self_loop:
            self.w_self = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.w_self_evolve = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.init_weight()

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
        # self loop message
        if self.self_loop:
            sl_msg = torch.mm(nodes_embed, self.w_self_evolve)
            sl_index = torch.unique(edges[:, 2])
            sl_msg[sl_index] = torch.mm(nodes_embed[sl_index], self.w_self)
        else:
            sl_msg = None
        # calculate message
        msg = torch.mm(nodes_embed[edges[:, 0]] + relation_embed[edges[:, 1]],
                       self.w_neighbor)
        # aggregate
        agg_index, msg_agg = GNN.gcn_aggregate(msg, edges[:, 2])
        # send message
        new_nodes_embed = nodes_embed.clone()
        new_nodes_embed[agg_index] += msg_agg
        if self.self_loop:
            new_nodes_embed = new_nodes_embed + sl_msg
        if self.active:
            torch.relu_(new_nodes_embed)
        new_nodes_embed = self.dropout(new_nodes_embed)
        return new_nodes_embed
