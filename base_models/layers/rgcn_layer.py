import torch
import torch.nn as nn
import math
from base_models.layers.gnn import GNN


class RGCNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_rels,
                 use_basis=False,
                 num_basis=1,
                 use_block=False,
                 num_block=1,
                 self_loop=True,
                 dropout_s=0,
                 dropout_o=0
                 ):
        super(RGCNLayer, self).__init__()
        self.use_basis = use_basis
        self.use_block = use_block
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_rels = num_rels
        self.self_loop = self_loop
        self.dropout_s = torch.nn.Dropout(p=dropout_s, inplace=True)
        self.dropout_o = torch.nn.Dropout(p=dropout_o, inplace=True)

        if use_basis:
            self.num_basis = num_basis
            self.v_weight = nn.Parameter(torch.Tensor(num_basis, input_dim, output_dim))
            self.weight = nn.Parameter(torch.Tensor(num_rels, num_basis))
        elif use_block:
            if input_dim % num_block != 0 or output_dim % num_block != 0:
                raise Exception('dimension do not match to the number of blocks')
            self.num_block = num_block
            self.b_weight = nn.Parameter(
                torch.Tensor(num_rels, num_block, input_dim // num_block, output_dim // num_block)
            )
        else:
            self.weight = nn.Parameter(torch.Tensor(num_rels, input_dim, output_dim))
        if self_loop:
            self.self_loop_weigt = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.weight_init()

    def weight_init(self):
        # self loop weight
        if self.self_loop:
            gain = nn.init.calculate_gain('relu')
            std = gain * math.sqrt(2 / (self.input_dim + self.output_dim))
            nn.init.normal_(self.self_loop_weigt, std=std, mean=0)
        # relation weight
        if self.use_basis:
            gain = nn.init.calculate_gain('relu')
            std = gain * math.sqrt(2 / (self.input_dim + self.output_dim))
            nn.init.normal_(self.v_weight, std=std, mean=0)
            nn.init.uniform_(self.weight, a=0, b=2 / self.num_basis)
        elif self.use_block:
            gain = nn.init.calculate_gain('relu')
            std = gain * math.sqrt(2 / (self.input_dim // self.num_block + self.output_dim // self.num_block))
            nn.init.normal_(self.b_weight, std=std, mean=0)
        else:
            gain = nn.init.calculate_gain('relu')
            std = gain * math.sqrt(2 / (self.input_dim + self.output_dim))
            nn.init.normal_(self.weight, std=std, mean=0)

    def get_relation_weight(self) -> torch.Tensor:
        if self.use_basis:
            res = torch.einsum('kx,xnm->knm', self.weight, self.v_weight)
        elif self.use_block:
            res = self.b_weight
        else:
            res = self.weight
        return res

    def forward(self,
                input_h,
                edges) -> torch.Tensor:
        """
        :param input_h: node embeddings, shape (num_nodes, input_dim)
        :param edges: list of triplets (src, rel, dst)
        :return: new node embeddings, shape (num_nodes, output_dim)
        """
        # separate triplets into src, rel, dst
        src, rel, dst = edges.transpose(0, 1)
        # calculate massage
        if self.use_block:
            msg = torch.matmul(
                input_h[src].view(-1, self.num_block, self.input_dim // self.num_block).unsqueeze(2),
                self.get_relation_weight()[rel]
            ).view(-1, self.output_dim)
        else:
            msg = torch.bmm(input_h[src].unsqueeze(1), self.get_relation_weight()[rel]).squeeze(1)
        # dropout
        msg = self.dropout_o(msg)
        # aggregate message
        dst_index, msg = GNN.gcn_aggregate(msg, dst)
        # self-loop message
        if self.self_loop:
            h = torch.mm(input_h, self.self_loop_weigt)
            h = self.dropout_s(h)
        # compose message
        h[dst_index] = h[dst_index].add_(msg)
        return h
