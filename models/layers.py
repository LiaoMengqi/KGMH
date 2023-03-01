import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, active='rrelu'):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param active:
        """
        super(GCNLayer, self).__init__()
        self.fc_self = nn.Linear(input_dim, output_dim, bias=bias)
        self.fc_aggregate = nn.Linear(input_dim, output_dim, bias=bias)
        if active == 'rrelu':
            self.active = nn.RReLU()
        elif active == 'sigmoid':
            self.active = nn.Sigmoid()
        elif active == 'tanh':
            self.active = nn.Tanh()

    def forward(self, node_embed, edge_embed, edge):
        """
        :param node_embed:Tensor, size=(num_node,input_dim)
        :param edge_embed: Tensor,size=(num_edge,input_dim)
        :param edge: Tensor, size=(num_edge, 3), with the format of (source node, edge, destination node)
        :return: the representation of node after aggregation
        """
        # self loop
        h = self.fc_self(node_embed)

        # calculate message
        message = node_embed[edge[:, 0]] + edge_embed[edge[:, 1]]
        message = self.fc_aggregate(message)

        # aggregate
        num_node = node_embed.shape[0]
        num_edge = edge.shape[0]
        des_unique, count = torch.unique(edge[:, 2], return_counts=True)

        index_en = csr_matrix((np.array(range(des_unique.shape[0]), dtype='int64'),
                               (des_unique, np.zeros(des_unique.shape[0], dtype='int64'))),
                              shape=(num_node, 1))
        index = torch.zeros(num_edge, node_embed.shape[1], dtype=torch.int64) + index_en[edge[:, 2]].todense()
        message = torch.zeros(des_unique.shape[0], node_embed.shape[1]).scatter_(0, index, message, reduce='add')
        message = message / (count.reshape(des_unique.shape[0], 1))

        # send message
        h[des_unique] = h[des_unique] + message

        return self.active(h)
