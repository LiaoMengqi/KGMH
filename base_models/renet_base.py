import torch.nn as nn


class ReNetBase(nn.Module):
    def __init__(self, num_entity, num_relation, seq_len):
        super(ReNetBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.seq_len = seq_len

    def forward(self):
        return
