import math

import torch
import torch.nn as nn


class TransEBase(nn.Module):
    def __init__(self, num_entity, num_relation, emb_dim, dtype=torch.float, margin=1.0, p_norm=1, c_e=2, c_r=1):
        super(TransEBase, self).__init__()
        self.p_norm = p_norm
        self.margin = margin
        self.c_e = c_e
        self.c_r = c_r
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

    def get_entity_embedding(self,
                             index):
        return self.entity_embedding(index)

    def get_relation_embedding(self,
                               index):
        return self.relation_embedding(index)

    def forward(self,
                edge: torch.Tensor):
        """
        Loss described in paper-Translating Embeddings for Modeling Multi-relational Data
        """
        return (self.entity_embedding(edge[:, 0]) + self.relation_embedding(edge[:, 1]) - self.entity_embedding(
            edge[:, 2])).norm(p=self.p_norm, dim=-1)
