import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransEBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 emb_dim,
                 margin=1.0,
                 p_norm=1,
                 c_e=2,
                 c_r=1):
        super(TransEBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.emb_dim = emb_dim
        self.p_norm = p_norm
        self.margin = margin
        self.c_e = c_e
        self.c_r = c_r
        self.entity_embedding = nn.Embedding(num_entity, emb_dim)
        self.relation_embedding = nn.Embedding(num_relation, emb_dim)
        self.weight_init()

    def weight_init(self):
        # bound = 6 / math.sqrt(self.emb_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        """with torch.no_grad():
            norm2 = self.relation_embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.relation_embedding.weight.copy_(self.relation_embedding.weight / norm2)
            norm2 = self.entity_embedding.weight.norm(p=2, dim=1, keepdim=True)
            self.entity_embedding.weight.copy_(self.entity_embedding.weight / norm2)"""

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
        h = F.normalize(self.entity_embedding(edge[:, 0]), 2, -1)
        r = F.normalize(self.relation_embedding(edge[:, 1]), 2, -1)
        t = F.normalize(self.entity_embedding(edge[:, 2]), 2, -1)
        return (h + r - t).norm(p=self.p_norm, dim=-1)
