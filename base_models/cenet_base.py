import torch
import torch.nn as nn
import torch.nn.functional as F


class CeNetBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 dim,
                 drop_prop=0.4,
                 lambdax=0.3,
                 alpha=0.1,
                 mode='soft'):
        super(CeNetBase, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.dim = dim
        self.drop_prop=drop_prop
        self.dropout = nn.Dropout(p=drop_prop)
        self.lambdax = lambdax
        self.alpha = alpha
        if mode not in ['solid','soft']:
            raise Exception('mode don\'t exist!')
        self.mode=mode

        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_relation, dim))
        self.entity_embeds = nn.Parameter(torch.zeros(num_entity, dim))

        self.linear_his = nn.Linear(dim * 2, dim)
        self.linear_nhis = nn.Linear(dim * 2, dim)
        self.linear_query = nn.Linear(dim * 3, dim)
        self.linear_freq = nn.Linear(num_entity, dim)
        self.bi_classifier = nn.Sequential(nn.Linear(dim * 3, dim * 3),
                                           nn.BatchNorm1d(dim * 3),
                                           nn.Dropout(0.4),
                                           nn.LeakyReLU(0.2),
                                           nn.Linear(dim * 3, 1),
                                           )

        self.weight_init()

    def weight_init(self):
        # embeddings
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))
        # Linear
        torch.nn.init.xavier_uniform_(self.linear_his.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.linear_nhis.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.linear_freq.weight, gain=nn.init.calculate_gain('relu'))

    def freeze_embed(self, freeze=False):
        self.rel_embeds.requires_grad_(freeze)
        self.entity_embeds.requires_grad_(freeze)
        self.linear_freq.requires_grad_(freeze)

    def forward(self, data_batched, history: torch.Tensor):
        """

        :param data_batched:Tensor (batch_size,3)
        :param history: Tensor (batch_size,num_entity)
        :return:
        """

        bias = history.clone()
        bias[bias > 0] = self.lambdax
        bias[bias == 0] = -self.lambdax
        simi_his = F.tanh(self.linear_his(
            self.dropout(
                torch.cat([self.entity_embeds[data_batched[:, 0]], self.rel_embeds[data_batched[:, 1]]], dim=1))))
        h_his = simi_his.mm(self.entity_embeds.T) + bias
        simi_nhis = F.tanh(self.linear_nhis(
            self.dropout(
                torch.cat([self.entity_embeds[data_batched[:, 0]], self.rel_embeds[data_batched[:, 1]]], dim=1))))
        h_nhis = simi_nhis.mm(self.entity_embeds.T) - bias
        return h_his, h_nhis
