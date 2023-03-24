import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_rels):
        super(RGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rels = num_rels
        self.weight = nn.Parameter(torch.Tensor(num_rels, input_dim, output_dim))
        self.self_loop_weigt = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def aggregate(self, message, des):
        des_unique, des_index, count = torch.unique(des, return_inverse=True, return_counts=True)
        message = torch.zeros(des_unique.shape[0], message.shape[1], dtype=message.dtype).scatter_add_(
            0, des_index.unsqueeze(1).expand_as(message), message)
        return des_unique, message / count.reshape(-1, 1)

    def forward(self, h, edges):
        """
        :param h: node embeddings, shape (num_nodes, input_dim)
        :param edges: list of triplets (src, rel, dst)
        :return: new node embeddings, shape (num_nodes, output_dim)
        """
        # separate triplets into src, rel, dst
        src, rel, dst = edges.transpose(0, 1)
        # gather node embeddings by indices
        src_h = h[src]
        # gather relation weights by indices
        weight = self.weight[rel]
        index_matrix = torch.zeros(h.shape[0], dtype=torch.long)
        index_matrix[dst] = torch.arange(dst.shape[0], dtype=torch.long)
        msg = torch.bmm(src_h.unsqueeze(1), weight).squeeze(1)
        # sort message corresponding to destination node
        msg = msg[index_matrix[dst]]
        # aggregate message
        dst_index, msg = self.aggregate(msg, dst)
        # self loop message passing
        out = torch.mm(h, self.self_loop_weigt)
        # compose message
        out[dst_index] = out[dst_index] + msg
        return out
