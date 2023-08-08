import torch
import torch.nn as nn


class CyGNetBase(nn.Module):
    def __init__(self,
                 num_entity,
                 num_relation,
                 h_dim,
                 alpha=0.5,
                 penalty=-100):
        super(CyGNetBase, self).__init__()
        self.reg_fact = 0.01
        self.alpha = alpha
        self.num_entity = num_entity
        self.num_relation=num_relation
        self.h_dim = h_dim
        self.unit_time_embed = nn.Parameter(torch.Tensor(1, h_dim))
        self.entity_embed = nn.Parameter(torch.Tensor(num_entity, h_dim))
        self.relation_embed = nn.Parameter(torch.Tensor(num_relation, h_dim))

        self.w_c = nn.Linear(h_dim * 3, self.num_entity)
        self.w_g = nn.Linear(h_dim * 3, self.num_entity)

        self.penalty = penalty
        self.weight_init()

    def nan_to_zero(self):
        with torch.no_grad():
            self.unit_time_embed.data = torch.nan_to_num(self.unit_time_embed.data)
            self.entity_embed.data = torch.nan_to_num(self.entity_embed.data)
            self.relation_embed.data = torch.nan_to_num(self.relation_embed.data)

    def weight_init(self):
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
        mask = (vocabulary == 0).type(vocabulary.dtype) * self.penalty
        score = score + mask
        # return nn.functional.softmax(score, dim=1)
        return nn.functional.log_softmax(score, dim=1)

    def generate_score(self,
                       e_embed,
                       r_embed,
                       t_embed) -> torch.Tensor:
        matrix = torch.cat([e_embed, r_embed, t_embed.expand(e_embed.shape)], dim=-1)
        score = self.w_g(matrix)
        # return nn.functional.softmax(score, dim=-1)
        return nn.functional.log_softmax(score, dim=1)

    def forward(self,
                edge,
                vocabulary,
                time_stamp,
                mode='obj') -> torch.Tensor:
        entity_embed = self.entity_embed[edge[:, 0]]
        relation_embed = self.relation_embed[edge[:, 1]]
        time_embed = self.get_time_embed(time_stamp)
        score_c = self.copy_score(entity_embed, relation_embed, time_embed, vocabulary)
        score_g = self.generate_score(entity_embed, relation_embed, time_embed)
        score = score_c * self.alpha + score_g * (1 - self.alpha)
        return score
