import torch
import torch.nn as nn

from base_models.layers.gnn import GNN


class WGCNLayer(nn.Module):
    def __init__(self,
                 num_relation,
                 input_dim,
                 output_dim,
                 bias=True):
        super(WGCNLayer, self).__init__()
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.relation_weight = nn.Parameter(torch.rand((num_relation, 1)))
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_weight()

    def init_weight(self):
        stdv = 1. / (self.output_dim ** 0.5)
        self.fc.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.fc.bias.data.uniform_(-stdv, stdv)

    def calculate_message(self,
                          src,
                          relation_weight):
        return self.fc(src * relation_weight)

    def forward(self,
                nodes_embed,
                edges):
        """
        :param nodes_embed: Tensor, the embedding of nodes, size=(num_node,input_dim)
        :param edges: Tensor, size=(num_edge, 3), with the format of (source node, edge, destination node)
        :return: new representation of nodes
        """
        h = self.fc(nodes_embed)
        message = self.calculate_message(nodes_embed[edges[:, 0]], self.relation_weight[edges[:, 1]])
        des, message = GNN.gcn_aggregate(message, edges[:, 2], normalize=None)
        h[des] = h[des] + message
        return h
