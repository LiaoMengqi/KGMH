import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channel,
                 kernel_length,
                 active='relu',
                 dropout=0.0,
                 bias=False):
        super(ConvTransEBase, self).__init__()
        self.input_dim = input_dim
        self.c = num_channel
        self.k = kernel_length
        self.dropout = torch.nn.Dropout(dropout)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.c)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)

        if kernel_length % 2 != 0:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2), int(kernel_length / 2), 0, 0))
        else:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2) - 1, int(kernel_length / 2), 0, 0))
        self.conv = nn.Conv2d(1, num_channel, (2, kernel_length))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_dim * num_channel, input_dim, bias=bias)

        if active == 'sigmoid':
            self.active = nn.Sigmoid()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            self.active = nn.ReLU()

    def forward(self,
                entity_ebd,
                relation_ebd,
                query,
                training=True):
        """
        :param entity_ebd:Tensor, size=(num_entity, input_dim)
        :param relation_ebd: Tensor, size=(num_relation, input_dim)
        :param query: LongTensor, size=(num_query,2)
        :return: Tensor,size=(num_query, num_entity), the item with the index of (i,j) in this tensor measures the
        possibility of entity j being the objective entity of query i.
        """
        x = torch.cat([entity_ebd.unsqueeze(dim=1)[query[:, 0]],
                       relation_ebd.unsqueeze(dim=1)[query[:, 1]]],
                      dim=1)
        x = self.bn0(x)
        if training:
            x = self.dropout(x)
        x.unsqueeze_(dim=1)
        x = self.conv(self.pad(x))
        x = self.bn1(x.squeeze())
        x = F.relu(x)
        if training:
            x = self.dropout(x)
        x = self.flat(x)
        x = self.fc(x)
        if training:
            x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        scores = torch.mm(x, entity_ebd.transpose(0, 1))
        return scores
