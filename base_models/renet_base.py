import torch.nn as nn
import torch
from base_models.rgcn_base import RGCNBase


class ReNetBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 hidden_dim,
                 dropout=0,
                 seq_len=10,
                 mode=0,
                 num_k=10):
        super(ReNetBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.seq_len = seq_len

    def init_weight(self, edge, ):
        pass

    def forward(self):
        return


class RGCNAggregator(nn.Module):
    def __init__(self,
                 num_entity,
                 num_rels,
                 input_dim,
                 h_dim,
                 dropout=0,
                 seq_len=10,
                 num_k=10,
                 aggr_mode='max'
                 ):
        super(RGCNAggregator, self).__init__()
        self.rgcn_aggr = RGCNAggregatorGlobal(input_dim,
                                              h_dim,
                                              num_rels,
                                              num_k,
                                              dropout)
    def forward(self):
        pass


class ReNetGlobalBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_rels,
                 input_dim,
                 h_dim,
                 dropout=0,
                 seq_len=10,
                 num_k=10,
                 aggr_mode='max'):
        super(ReNetGlobalBase, self).__init__()
        self.seq_len = seq_len
        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, input_dim))
        self.rgcn_aggr = RGCNAggregatorGlobal(input_dim,
                                              h_dim,
                                              num_rels,
                                              num_k,
                                              dropout)
        self.gru = nn.GRU(h_dim,
                          h_dim,
                          batch_first=True)
        self.linear = nn.Linear(h_dim, num_entity)

    def weight_init(self):
        nn.init.uniform_(self.entity_embed)

    def h2input(self,
                h: torch.Tensor):
        """
        translate a tensor to input of RNN
        :param h: a two dimension tensor
        :return: a packed variable length sequence
        """
        total_time = h.shape[0]
        seq_list = []
        seq_len_list = []
        target_time = []
        for i in reversed(range(total_time - 1)):
            if i < self.seq_len:
                seq_list.append(torch.arange(0, i + 1, dtype=torch.long, device=h.device))
                seq_len_list.append(i + 1)
                target_time.append(i + 1)
            else:
                seq_list.append(torch.arange(i - self.seq_len + 1, i + 1, dtype=torch.long, device=h.device))
                seq_len_list.append(self.seq_len)
                target_time.append(i + 1)
        seq_len_list = torch.LongTensor(seq_len_list)
        embed_seq_tensor = torch.zeros(len(seq_list), self.seq_len, h.shape[1], device=h.device)
        for i, seq in enumerate(seq_list):
            for j, t in enumerate(seq):
                embed_seq_tensor[i, j, :] = h[t]
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               seq_len_list,
                                                               batch_first=True)
        return packed_input, torch.LongTensor(target_time).to(h.device)

    def forward(self, edges: list):
        global_h = self.rgcn_aggr(edges, self.entity_embed)
        input_packed, target_index = self.h2input(global_h)
        _, hidden = self.gru(input_packed)
        hidden = hidden.squeeze()
        score = self.linear(hidden)
        return score, target_index

    def get_global_embed(self, edges):
        if len(edges) > self.seq_len:
            edges = edges[-self.seq_len:-1]
        global_h = self.rgcn_aggr(edges, self.entity_embed)
        input_packed, target_index = self.h2input(global_h)
        _, hidden = self.gru(input_packed)
        return hidden, target_index


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
        """
        get global representation of temporal knowledge graph
        :param edges: a list, each element is a set of edges from a static knowledge graph
        :param entity_embed: embedding of nodes
        :return: global representation
        """
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
        return self.dropout(torch.cat(res_list, dim=0))
