import torch.nn as nn
import torch


class ReNetBase(nn.Module):
    def __init__(self, num_entity, num_relation, seq_len):
        super(ReNetBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.seq_len = seq_len

    def forward(self):
        return


class ReNetGlobal(nn.Module):
    def __init__(self, num_entity, h_dim, num_rels, dropout=0, seq_len=10, num_k=10, maxpool=1):
        super(ReNetGlobal, self).__init__()
        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, h_dim))

    def weight_init(self):
        nn.init.xavier_uniform_(self.entity_embed,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self):
        return
