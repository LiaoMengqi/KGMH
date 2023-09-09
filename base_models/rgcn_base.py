import torch
import torch.nn as nn
from base_models.layers.rgcn_layer import RGCNLayer
import torch.nn.functional as F


class RGCNBase(nn.Module):
    def __init__(self,
                 dims: list,
                 num_relation: int,
                 num_entity: int,
                 use_basis=False,
                 num_basis=10,
                 use_block=False,
                 num_block=10,
                 dropout_s=0,
                 dropout_o=0,
                 inverse=True
                 ):
        super(RGCNBase, self).__init__()
        self.dims = dims
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.use_basis = use_basis
        self.num_basis = num_basis
        self.use_block = use_block
        self.num_block = num_block
        self.dropout_s = dropout_s
        self.dropout_o = dropout_o
        self.inverse = inverse

        self.layers = nn.ModuleList()
        if len(dims) < 2:
            raise Exception('At least have two dimension!')
        self.inverse = inverse
        num_rela_expand = self.num_relation * 2 if self.inverse else self.num_relation
        for i in range(len(dims) - 1):
            self.layers.append(RGCNLayer(dims[i],
                                         dims[i + 1],
                                         num_rela_expand,
                                         use_basis=use_basis,
                                         num_basis=num_basis,
                                         use_block=use_block,
                                         num_block=num_block,
                                         dropout_s=dropout_s,
                                         dropout_o=dropout_o
                                         )
                               )

        self.num_layer = len(dims) - 1
        self.entity_embed = nn.Embedding(num_entity, embedding_dim=dims[0])
        self.wight_init()

    def wight_init(self):
        pass

    def forward(self,
                edges: torch.Tensor,
                h_input=None):
        # initial input representation
        if h_input is None:
            h_input = self.entity_embed.weight

        # return representation of the last layer

        for i in range(self.num_layer):
            h_input = F.relu(self.layers[i](h_input, edges))
        return h_input
