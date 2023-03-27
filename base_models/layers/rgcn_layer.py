import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_rels, basis=False, b=10):
        super(RGCNLayer, self).__init__()
        self.basis = basis
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rels = num_rels
        if basis:
            self.b = b
            self.v_weight = nn.Parameter(torch.Tensor(b, input_dim, output_dim))
            self.weight = nn.Parameter(torch.Tensor(num_rels, b))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_rels, input_dim, output_dim))
        self.self_loop_weigt = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform_(self.self_loop_weigt, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        if self.basis:
            nn.init.uniform_(self.weight)
            nn.init.xavier_normal_(self.v_weight)
        else:
            nn.init.xavier_normal_(self.weight)
        return

    def get_relation_weight(self, index: torch.Tensor) -> torch.Tensor:
        if self.basis:
            res = torch.einsum('kx,xnm->knm', self.weight[index], self.v_weight)
        else:
            res = self.weight[index]
        return res

    def aggregate(self, message, des):
        des_unique, des_index, count = torch.unique(des, return_inverse=True, return_counts=True)
        message = torch.zeros(des_unique.shape[0], message.shape[1], dtype=message.dtype,
                              device=message.device).scatter_add_(
            0, des_index.unsqueeze(1).expand_as(message), message)
        return des_unique, message / count.reshape(-1, 1)

    def forward(self, input_h, edges):
        """
        :param input_h: node embeddings, shape (num_nodes, input_dim)
        :param edges: list of triplets (src, rel, dst)
        :return: new node embeddings, shape (num_nodes, output_dim)
        """
        # separate triplets into src, rel, dst
        src, rel, dst = edges.transpose(0, 1)
        # gather node embeddings by indices
        src_h = input_h[src]
        # gather relation weights by indices
        weight = self.get_relation_weight(rel)
        index_matrix = torch.zeros(input_h.shape[0], dtype=torch.long, device=weight.device)
        index_matrix[dst] = torch.arange(dst.shape[0], dtype=torch.long,device=weight.device)
        msg = torch.bmm(src_h.unsqueeze(1), weight).squeeze(1)
        # sort message corresponding to destination node
        msg = msg[index_matrix[dst]]
        # aggregate message
        dst_index, msg = self.aggregate(msg, dst)
        # self loop message passing
        output_h = torch.mm(input_h, self.self_loop_weigt)
        # compose message
        output_h[dst_index] = output_h[dst_index] + msg
        return output_h
