import torch
import torch.nn as nn
import torch.nn.functional as F

from base_models.layers.gcn_layer import GCNLayer


class GCNBase(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=None):
        super(GCNBase, self).__init__()
        self.layers = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = []
        all_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(all_dims) - 1):
            self.layers.append(GCNLayer(all_dims[i], all_dims[i + 1]))
        self.num_layer = len(all_dims - 1)

    def forward(self,
                node_presentation,
                edges):
        """
        :param node_presentation:Tensor, size=(num_nodes,input_dim)
        :param edges: LongTensor, size=(num_edges,2)
        :return: new presentations of nodes
        """
        if isinstance(edges, list):
            # edges is a list of temporal knowledge graphs
            for i in range(self.num_layer - 1):
                node_presentation = F.relu(self.layers[i](node_presentation, edges[i]))
            # The last layers with no active function
            node_presentation = self.layers[-1](node_presentation, edges[-1])
        else:
            for i in range(self.num_layer - 1):
                node_presentation = F.relu(self.layers[i](node_presentation, edges))
            node_presentation = self.layers[-1](node_presentation, edges)
        return node_presentation
