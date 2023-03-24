import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class WGCNLayer(nn.Module):
    def __init__(self, num_relation, input_dim, output_dim, active='relu', bias=False, dtype=torch.float):
        super(WGCNLayer, self).__init__()
        self.num_relation = num_relation
        self.input_dim = input_dim
        self.dtype = dtype
        self.output_dim = output_dim
        self.relation_weight = nn.Parameter(torch.rand((num_relation, 1), dtype=dtype))
        self.fc = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)
        if active == 'sigmoid':
            self.active = nn.Sigmoid()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            self.active = nn.ReLU()

    def calculate_message(self, src, relation_weight):
        return self.fc(src * relation_weight)

    def aggregate(self, message, num_node, des):
        des_unique, count = torch.unique(des, return_counts=True)
        index_matrix = csr_matrix((np.array(range(des_unique.shape[0]), dtype='int64'),
                                   (des_unique, np.zeros(des_unique.shape[0], dtype='int64'))),
                                  shape=(num_node, 1))
        index = torch.zeros(message.shape[0], message.shape[1], dtype=torch.int64) + index_matrix[des].todense()
        message = torch.zeros(des_unique.shape[0], self.output_dim, dtype=self.dtype).scatter_(0, index, message,
                                                                                               reduce='add')
        return des_unique, message

    def forward(self, nodes_embed, edges):
        """
        :param nodes_embed: Tensor, the embedding of nodes, size=(num_node,input_dim)
        :param edges: Tensor, size=(num_edge, 3), with the format of (source node, edge, destination node)
        :return: new representation of nodes
        """
        h = self.fc(nodes_embed)
        message = self.calculate_message(nodes_embed[edges[:, 0]], self.relation_weight[edges[:, 1]])
        des_index, message = self.aggregate(message, nodes_embed.shape[0], edges.shape[0], edges[:, 2])
        h[des_index] = h[des_index] + message
        return self.active(h)
