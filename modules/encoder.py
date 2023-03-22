import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import GCNLayer


class TransE(nn.Module):
    def __init__(self, num_entity, num_relation, emb_dim, dtype=torch.float, margin=1.0, p_norm=1, c=1):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.margin = margin
        self.c = c
        self.num_entity = num_entity
        self.emb_dim = emb_dim
        self.entity_embedding = nn.Embedding(num_entity, emb_dim, dtype=dtype)
        self.relation_embedding = nn.Embedding(num_relation, emb_dim, dtype=dtype)
        self.weight_init()

    def weight_init(self):
        bound = 6 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.entity_embedding.weight, a=-bound, b=bound)
        nn.init.uniform_(self.relation_embedding.weight, a=-bound, b=bound)
        with torch.no_grad():
            norm2 = self.relation_embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.relation_embedding.weight.copy_(self.relation_embedding.weight / norm2)
            norm2 = self.entity_embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.entity_embedding.weight.copy_(self.entity_embedding.weight / norm2)

    def norm_weight(self):
        with torch.no_grad():
            norm2 = self.entity_embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.entity_embedding.weight.copy_(self.entity_embedding.weight / norm2)

    def get_entity_embedding(self, index):
        return self.entity_embedding(index)

    def get_relation_embedding(self, index):
        return self.relation_embedding(index)

    def forward(self, edge: torch.Tensor):
        """
        Loss described in paper-Translating Embeddings for Modeling Multi-relational Data
        """
        # x = self.entity_embedding(edge[:, 0]).norm(p=2, dim=-1)
        # y = self.relation_embedding(edge[:, 1]).norm(p=2, dim=-1)
        return (self.entity_embedding(edge[:, 0]) + self.relation_embedding(edge[:, 1]) - self.entity_embedding(
            edge[:, 2])).norm(p=self.p_norm, dim=1)

    def predict(self, sub, rela):
        hr = self.entity_embedding(sub) + self.relation_embedding(rela)
        t = self.entity_embedding(torch.LongTensor(range(self.num_entity)).to(hr.device))
        return (hr.unsqueeze(1) - t.unsqueeze(0)).norm(p=self.p_norm, dim=-1)

    def scale_loss(self, embed):
        return F.relu(embed.norm(p=2, dim=1) - 1).sum() / embed.shape[0]

    def loss(self, pos_edge, nag_edge):
        # self.norm_weight()
        pos_dis = self(pos_edge)
        nag_dis = self(nag_edge)
        loss = F.relu(self.margin + pos_dis - nag_dis).sum()
        rela_scale_loss = self.scale_loss(self.relation_embedding(pos_edge[:, 1]))
        entity = torch.cat([pos_edge[:, 0], pos_edge[:, 2], nag_edge[:, 0], nag_edge[:, 2]])
        entity_scale_loss = self.scale_loss(self.entity_embedding(entity))
        loss = loss + self.c * (rela_scale_loss + entity_scale_loss)
        return loss


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = []
        all_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(all_dims) - 1):
            self.layers.append(GCNLayer(all_dims[i], all_dims[i + 1]))
        self.num_layer = len(all_dims - 1)

    def forward(self, node_presentation, edges):
        """
        :param node_presentation:Tensor, size=(num_nodes,input_dim)
        :param edges: LongTensor, size=(num_edges,2)
        :return: new presentations of nodes
        """
        if isinstance(edges, list):
            # edges is a list of temporal knowledge graphs
            for i in range(self.num_layer - 1):
                node_presentation = F.relu(self.layers[i](node_presentation, edges[i]))
            # The last layer with no active function
            node_presentation = self.layers[-1](node_presentation, edges[-1])
        else:
            for i in range(self.num_layer - 1):
                node_presentation = F.relu(self.layers[i](node_presentation, edges))
            node_presentation = self.layers[-1](node_presentation, edges)
        return node_presentation
