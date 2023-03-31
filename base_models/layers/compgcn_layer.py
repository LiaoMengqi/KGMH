import torch
import torch.nn as nn


class CompGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_rela, dtype=torch.float):
        super(CompGCNLayer, self).__init__()
        self.output_dim = output_dim
        self.num_rela = num_rela
        self.dtype = dtype
        self.W_o = nn.Linear(input_dim, output_dim, bias=False)
        self.W_i = nn.Linear(input_dim, output_dim, bias=False)
        self.W_s = nn.Linear(input_dim, output_dim, bias=False)
        self.W_r = nn.Linear(input_dim, output_dim, bias=False)

    def composition(self,
                    node_embed,
                    rela_embed,
                    mode='add'):
        if mode == 'add':
            res = node_embed + rela_embed
        elif mode == 'sub':
            res = node_embed - rela_embed
        elif mode == 'mult':
            res = node_embed * rela_embed
        else:
            res = None
        return res

    def aggregate(self, message, des):
        des_unique, des_index = torch.unique(des, return_inverse=True)
        message = torch.zeros(des_unique.shape[0], message.shape[1], dtype=self.dtype).scatter_add_(
            0, des_index.unsqueeze(1).expand_as(message), message)
        return des_unique, message

    def forward(self,
                node_embed,
                rela_embed,
                edges,
                mode='add'):
        """
        :param node_embed:
        :param rela_embed:
        :param edges: LongTensor, including the original edge and reversed edge
        :param mode: Method to composite representations of relations and nodes
        :return:
        """
        # self loop
        h_v = self.W_i(self.composition(node_embed, rela_embed[self.num_rela * 2], mode))

        # original edges
        index = edges[:, 1] < self.num_rela
        src = edges[index][:, 0]
        rela = edges[index][:, 1]
        des = edges[index][:, 2]
        index_matrix = torch.zeros(node_embed.shape[0], dtype=torch.long)
        index_matrix[des] = torch.arange(des.shape[0], dtype=torch.long)
        message = self.W_o(self.composition(node_embed[src], rela_embed[rela]))
        message = message[index_matrix[des]]
        des_index, message = self.aggregate(message, des)
        h_v[des_index] = h_v[des_index] + message

        # reversed edges
        index = edges[:, 1] >= self.num_rela
        src = edges[index][:, 0]
        rela = edges[index][:, 1]
        des = edges[index][:, 2]
        index_matrix[des] = torch.arange(des.shape[0], dtype=torch.long)
        message = self.W_s(self.composition(node_embed[src], rela_embed[rela]))
        message = message[index_matrix[des]]
        des_index, message = self.aggregate(message, des)
        h_v[des_index] = h_v[des_index] + message

        # update relation representation
        h_r = self.W_r(rela_embed)
        return h_v, h_r
