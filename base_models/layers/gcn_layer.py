import torch
import torch.nn as nn
from base_models.layers.gnn import GNN


class GCNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(GCNLayer, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self,
                node_embed,
                edges):
        """
        :param node_embed: Tensor, size=(num_node, input_dim), Embeddings of nodes
        :param edge_index: LongTensor ,size=(num_edge, 2), source nodes and destination nodes
        :return:
        """
        h = self.lin(node_embed)
        num_entity = node_embed.shape[0]
        a = GNN.edges2adj(edges, num_entity)

        indices = torch.arange(num_entity, device=node_embed.device)
        values = torch.ones(num_entity, device=node_embed.device)
        i = torch.sparse_coo_tensor(torch.cat([indices.unsqueeze(0), indices.unsqueeze(0)], dim=0), values,
                                    (num_entity, num_entity), device=node_embed.device)

        out_degree = torch.pow(GNN.cal_out_degree(a + i), -0.5)
        d = torch.sparse_coo_tensor(torch.cat([indices.unsqueeze(0), indices.unsqueeze(0)], dim=0), out_degree.values(),
                                    (num_entity, num_entity), device=node_embed.device)
        h = torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(d, a + i), d), h)
        return h
