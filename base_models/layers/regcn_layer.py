import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class REGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, active='rrelu', dtype=torch.float64):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param active:
        """
        super(REGCNLayer, self).__init__()
        self.dtype = dtype
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_self = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        self.fc_aggregate = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        if active == 'rrelu':
            self.active = nn.RReLU()
        elif active == 'sigmoid':
            self.active = nn.Sigmoid()
        elif active == 'tanh':
            self.active = nn.Tanh()

    def calculate_message(self, src, rela):
        return self.fc_aggregate(src + rela)

    def aggregate(self, message, num_node, des):
        des_unique, count = torch.unique(des, return_counts=True)
        index_matrix = csr_matrix((np.array(range(des_unique.shape[0]), dtype='int64'),
                                   (des_unique, np.zeros(des_unique.shape[0], dtype='int64'))),
                                  shape=(num_node, 1))
        index = torch.zeros(message.shape[0], self.output_dim, dtype=torch.int64) + index_matrix[des].todense()
        message = torch.zeros(des_unique.shape[0], self.output_dim, dtype=self.dtype).scatter_(0, index, message,
                                                                                               reduce='add')
        return des_unique, message / (count.reshape(des_unique.shape[0], 1))

    def forward(self, nodes_embed, edges_embed, edges):
        """
        :param nodes_embed:Tensor, size=(num_node,input_dim)
        :param edges_embed: Tensor,size=(num_edge,input_dim)
        :param edge: Tensor, size=(num_edge, 3), with the format of (source node, edge, destination node)
        :return: the representation of node after aggregation
        """
        # self loop
        h = self.fc_self(nodes_embed)
        # calculate message
        message = self.calculate_message(nodes_embed[edges[:, 0]], edges_embed[edges[:, 1]])
        # aggregate
        des_index, message = self.aggregate(message, nodes_embed.shape[0], edges[:, 2])
        # send message
        h[des_index] = h[des_index] + message
        return self.active(h)