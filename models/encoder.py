import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, num_entity, num_relation, emb_dim, max_norm=None, norm_type=2):
        super(TransE, self).__init__()
        if max_norm:
            self.entity_embedding = nn.Embedding(num_entity, emb_dim, max_norm=max_norm, norm_type=norm_type)
            self.relation_embedding = nn.Embedding(num_relation, emb_dim, max_norm=max_norm, norm_type=norm_type)
        else:
            self.entity_embedding = nn.Embedding(num_entity, emb_dim)
            self.relation_embedding = nn.Embedding(num_relation, emb_dim)

    def forward(self, head=None, relation=None, tail=None):
        """
        :param head: LongTensor
        :param relation: LongTensor
        :param tail: LongTensor
        :return: embeddings of node or relation
        """
        if head:
            head_emb = self.entity_embedding(head)
        else:
            head_emb = None
        if relation:
            tail_emb = self.entity_embedding(tail)
        else:
            tail_emb = None
        if relation:
            relation_emb = self.relation_embedding(relation)
        else:
            relation_emb = None
        return head_emb, relation_emb, tail_emb
