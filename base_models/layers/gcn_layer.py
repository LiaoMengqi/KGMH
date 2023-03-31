import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self,
                node_embed,
                edges):
        """
        :param node_embed: Tensor, size=(num_node, input_dim), Embeddings of nodes
        :param edge_index: LongTensor ,size=(num_edge, 2), source nodes and destination nodes
        :return:
        """
        return

    def message(self,
                x_j,
                norm):
        # message passing
        return norm.view(-1, 1) * x_j

    def update(self,
               aggr_out):
        # update presentation of nodes
        return self.lin(aggr_out)
