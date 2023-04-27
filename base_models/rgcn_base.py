import torch
import torch.nn as nn
from base_models.layers.rgcn_layer import RGCNLayer
import torch.nn.functional as F


class RGCNBase(nn.Module):
    def __init__(self,
                 dim_list: list,
                 num_relation: int,
                 num_entity=None,
                 use_basis=False,
                 num_basis=10,
                 use_block=False,
                 num_block=10,
                 init_embed=True,
                 dropout_s=None,
                 dropout_o=None
                 ):
        super(RGCNBase, self).__init__()
        self.layers = nn.ModuleList()
        self.num_relation = num_relation
        self.dims = dim_list
        self.init_embed = init_embed
        if len(dim_list) < 2:
            raise Exception('At least have two dimension!')
        for i in range(len(dim_list) - 1):
            self.layers.append(RGCNLayer(dim_list[i],
                                         dim_list[i + 1],
                                         num_relation,
                                         use_basis=use_basis,
                                         num_basis=num_basis,
                                         use_block=use_block,
                                         num_block=num_block,
                                         dropout_s=dropout_s,
                                         dropout_o=dropout_o
                                         )
                               )
        self.num_layer = len(dim_list) - 1

        if not init_embed:
            self.entity_embed = None
        else:
            self.num_entity = num_entity
            if num_entity is None:
                raise Exception
            self.entity_embed = nn.Embedding(num_entity, embedding_dim=dim_list[0])
            self.wight_init()

    def wight_init(self):
        pass

    def forward(self,
                edges: torch.Tensor,
                h_input=None,
                training=True):
        # initial input representation
        if h_input is None:
            if not self.init_embed:
                raise Exception
            h_input = self.entity_embed.weight

        # return representation of the last layer
        for i in range(self.num_layer):
            h_input = F.relu(self.layers[i](h_input, edges))
        return h_input
