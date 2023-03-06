import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCNLayer


class TransE(nn.Module):
    def __init__(self, num_entity, num_relation, emb_dim, max_norm=None, norm_type=2, dtype=torch.float, margin=1.0):
        super(TransE, self).__init__()
        self.margin = margin
        if max_norm is not None:
            self.entity_embedding = nn.Embedding(num_entity, emb_dim, dtype=dtype, max_norm=max_norm,
                                                 norm_type=norm_type)
            self.relation_embedding = nn.Embedding(num_relation, emb_dim, dtype=dtype)
        else:
            self.entity_embedding = nn.Embedding(num_entity, emb_dim, dtype=dtype)
            self.relation_embedding = nn.Embedding(num_relation, emb_dim, dtype=dtype)

    def forward(self, head=None, relation=None, tail=None):
        """
        :param head: LongTensor
        :param relation: LongTensor
        :param tail: LongTensor
        :return: embeddings of node or relation
        """
        head_emb = None
        relation_emb = None
        tail_emb = None
        if head is not None:
            head_emb = self.entity_embedding(head)
        if relation is not None:
            relation_emb = self.relation_embedding(relation)
        if relation is not None:
            tail_emb = self.entity_embedding(tail)
        return head_emb, relation_emb, tail_emb

    def calculate_loss(self, head_emb, relation_emb, tail_emb, positive_edge, negative_edge):
        """
        Loss described in paper-Translating Embeddings for Modeling Multi-relational Data
        """
        loss = self.margin + torch.abs(
            head_emb[positive_edge[0]] + relation_emb[positive_edge[1]] - tail_emb[positive_edge[2]])
        loss = loss - torch.abs(
            head_emb[negative_edge[0]] + relation_emb[negative_edge[1]] - tail_emb[negative_edge[2]])
        loss = F.relu(torch.sum(loss, dim=1))
        return loss.sum()


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

