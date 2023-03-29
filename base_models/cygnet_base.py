import torch
import torch.nn as nn


class CyGNetBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 h_dim,
                 alpha=0.5):
        super(CyGNetBase, self).__init__()

        self.alpha = alpha
        self.num_entity = num_entity

        self.unit_time_embed = nn.Parameter(torch.Tensor(1, h_dim))
        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, h_dim))
        self.relation_embed = nn.Parameter(torch.Tensor(num_relation, h_dim))

        self.w_c = nn.Linear(h_dim * 3, self.num_entity)
        self.w_g = nn.Linear(h_dim * 3, self.num_entity)

        self.penalty_factor = 100
        self.weight_init()

    def weight_init(self):
        # 非权重，是嵌入，这样初始化有问题
        nn.init.xavier_uniform_(self.entity_embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.unit_time_embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embed, gain=nn.init.calculate_gain('relu'))

    def get_time_embed(self,
                       time_stamp) -> torch.Tensor:
        return self.unit_time_embed * time_stamp

    def copy_score(self,
                   e_embed: torch.Tensor,
                   r_embed: torch.Tensor,
                   t_embed: torch.Tensor,
                   vocabulary: torch.Tensor) -> torch.Tensor:
        matrix = torch.cat([e_embed, r_embed, t_embed.expand(e_embed.shape)], dim=-1)
        score = nn.functional.tanh(self.w_c(matrix))
        mask = vocabulary * self.penalty_factor
        return nn.functional.softmax(score + mask, dim=-1)

    def generate_score(self,
                       e_embed,
                       r_embed,
                       t_embed) -> torch.Tensor:
        matrix = torch.cat([e_embed, r_embed, t_embed.expand(e_embed.shape)], dim=-1)
        score = self.w_g(matrix)
        return nn.functional.softmax(score, dim=-1)

    def forward(self,
                edge,
                vocabulary,
                time_stamp,
                mode='obj') -> torch.Tensor:
        entity_embed = self.entity_embed[edge[:, 0]]
        relation_embed = self.relation_embed[edge[:, 1]]
        time_embed = self.get_time_embed(time_stamp)
        score = self.copy_score(entity_embed, relation_embed, time_embed, vocabulary) * self.alpha
        score = score + self.generate_score(entity_embed, relation_embed, time_embed) * (1 - self.alpha)
        return score
