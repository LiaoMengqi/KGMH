import torch
import torch.nn as nn
from base_models.layers.rgcn_layer import RGCNLayer
import torch.nn.functional as F


class RGCNBase(nn.Module):
    def __init__(self, dim_list: list, num_relation: int, num_entity: int, basis=False, b=10, input_embed=False):
        super(RGCNBase, self).__init__()
        self.layers = nn.ModuleList()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.dims = dim_list
        self.input_embed = input_embed
        if len(dim_list) < 2:
            raise Exception('At least have two dimension!')
        for i in range(len(dim_list) - 1):
            self.layers.append(RGCNLayer(dim_list[i], dim_list[i + 1], num_relation, basis=basis, b=b))
        self.num_layer = len(dim_list) - 1
        self.entity_embed = nn.Embedding(num_entity, embedding_dim=dim_list[0])

        if input_embed:
            self.entity_embed = None
        else:
            self.entity_embed = nn.Embedding(num_entity, embedding_dim=dim_list[0])
            self.wight_init()

    def wight_init(self):
        pass

    def forward(self,
                edges: torch.Tensor,
                h_input=None) -> torch.Tensor:
        if not self.input_embed:
            h_input = self.entity_embed.weight

        for i in range(self.num_layer - 1):
            h_input = F.leaky_relu(self.layers[i](h_input, edges), negative_slope=0.2)
        h_output = self.layers[-1](h_input, edges)
        return h_output
