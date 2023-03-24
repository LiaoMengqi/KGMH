import torch.nn as nn


class ReNet(nn.Module):
    def __init__(self, num_entity, num_relation, seq_len):
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.seq_len = seq_len

        super(ReNet, self).__init__()

    def forward(self):
        return
