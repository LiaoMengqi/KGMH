import torch.nn as nn
import torch
from base_models.regcn_base import RGCNBase


class ReNetBase(nn.Module):
    def __init__(self, num_entity, num_relation, seq_len):
        super(ReNetBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.seq_len = seq_len

    def forward(self):
        return


class ReNetGlobal(nn.Module):
    def __init__(self,
                 num_entity,
                 input_dim,
                 h_dim,
                 num_rels,
                 dropout=0,
                 seq_len=10,
                 num_k=10,
                 aggr_mode='max'):
        super(ReNetGlobal, self).__init__()
        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, h_dim))
        self.rgcn_aggr = RGCNAggregatorGlobal(input_dim,
                                              h_dim,
                                              num_rels,
                                              num_k,
                                              dropout)
        self.gru = nn.GRU(h_dim,
                          h_dim,
                          batch_first=True)

    def weight_init(self):
        nn.init.uniform_(self.entity_embed)

    def h2input(self,
                h: torch.Tensor):
        """
        translate a tensor to input of RNN
        :param h: a two dimension tensor
        :return:
        """
        embed_seq_tensor = h
        len_non_zero = h
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        return packed_input

    def forward(self, edges: list):
        global_h = self.rgcn_aggr(edges, self.entity_embed)
        input_packed = self.h2input(global_h)
        res = self.gru(input_packed)
        return res


class RGCNAggregatorGlobal(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_relation: int,
                 num_block: int,
                 dropout=0,
                 mode='max'):
        super(RGCNAggregatorGlobal, self).__init__()
        # RGCNs with two layers
        self.rgcn = RGCNBase([input_dim, output_dim, output_dim],
                             num_relation,
                             use_block=True,
                             num_block=num_block,
                             init_embed=False,
                             dropout_o=dropout,
                             dropout_s=dropout)
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                edges: list,
                entity_embed: torch.Tensor) -> torch.Tensor:
        res_list = []
        for i in range(len(edges)):
            res = self.rgcn(edges[i], entity_embed)
            if self.mode == 'max':
                res, _ = torch.max(res, dim=0)
                res_list.append(res.unsqueeze(0))
            elif self.mode == 'mean':
                res = res.mean(dim=0)
                res_list.append(res.unsqueeze(0))
            else:
                raise Exception
        return torch.cat(res_list, dim=0)
