import torch
import torch.nn as nn


class Conv_TransEBase(nn.Module):
    def __init__(self, input_dim, num_channel, kernel_length, active='relu', dtype=torch.float64, bias=False):
        super(Conv_TransEBase, self).__init__()
        self.input_dim = input_dim
        self.c = num_channel
        self.k = kernel_length
        self.dtype = dtype
        if kernel_length % 2 != 0:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2), int(kernel_length / 2), 0, 0))
        else:
            self.pad = nn.ZeroPad2d((int(kernel_length / 2) - 1, int(kernel_length / 2), 0, 0))
        self.conv = nn.Conv2d(1, num_channel, (2, kernel_length), dtype=dtype)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(input_dim * num_channel, input_dim, bias=bias, dtype=dtype)

        if active == 'sigmoid':
            self.active = nn.Sigmoid()
        elif active == 'tanh':
            self.active = nn.Tanh()
        else:
            self.active = nn.ReLU()

    def forward(self, entity_ebd, relation_ebd, query):
        """
        :param entity_ebd:Tensor, size=(num_entity, input_dim)
        :param relation_ebd: Tensor, size=(num_relation, input_dim)
        :param query: LongTensor, size=(num_query,2)
        :return: Tensor,size=(num_query, num_entity), the item with the index of (i,j) in this tensor measures the
        possibility of entity j being the objective entity of query i.
        """
        entity_ebd = entity_ebd.unsqueeze(dim=1)
        relation_ebd = relation_ebd.unsqueeze(dim=1)
        matrix = torch.cat([entity_ebd[query[:, 0]], relation_ebd[query[:, 1]]], dim=1)
        matrix.unsqueeze_(dim=1)
        conv_out = self.flat(self.conv(self.pad(matrix)))
        linear_out = self.active(self.fc(conv_out))
        linear_out.unsqueeze_(dim=1)
        entity_ebd.squeeze_(dim=1)
        scores = torch.sum(entity_ebd * linear_out, dim=2)
        return nn.functional.sigmoid(scores)
