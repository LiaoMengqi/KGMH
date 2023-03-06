import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.sparse import csr_matrix
import numpy as np



class RGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, active='rrelu', dtype=torch.float64):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param active:
        """
        super(RGCNLayer, self).__init__()
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

    def aggregate(self, message, num_node, num_edge, des):
        des_unique, count = torch.unique(des, return_counts=True)
        index_matrix = csr_matrix((np.array(range(des_unique.shape[0]), dtype='int64'),
                                   (des_unique, np.zeros(des_unique.shape[0], dtype='int64'))),
                                  shape=(num_node, 1))
        index = torch.zeros(num_edge, self.output_dim, dtype=torch.int64) + index_matrix[des].todense()
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
        des_index, message = self.aggregate(message, nodes_embed.shape[0], edges.shape[0], edges[:, 2])
        # send message
        h[des_index] = h[des_index] + message
        return self.active(h)


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

    def aggregate(self, message, num_node, num_edge, des):
        des_unique, count = torch.unique(des, return_counts=True)
        index_matrix = csr_matrix((np.array(range(des_unique.shape[0]), dtype='int64'),
                                   (des_unique, np.zeros(des_unique.shape[0], dtype='int64'))),
                                  shape=(num_node, 1))
        index = torch.zeros(num_edge, self.output_dim, dtype=torch.int64) + index_matrix[des].todense()
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



class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, node_embed, edges):
        """
        :param node_embed: Tensor, size=(num_node, input_dim), Embeddings of nodes
        :param edge_index: LongTensor ,size=(num_edge, 2), source nodes and destination nodes
        :return:
        """
        # self loop
        edges, _ = add_self_loops(edges, num_nodes=node_embed.size(0))
        deg = degree(edges[0], node_embed.size(0), dtype=node_embed.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        # normalization coefficient
        norm = deg_inv_sqrt[edges[0]] * deg_inv_sqrt[edges[1]]
        # propagate message
        return self.propagate(edges, x=node_embed, norm=norm)

    def message(self, x_j, norm):
        # message passing
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # update presentation of nodes
        return self.lin(aggr_out)


