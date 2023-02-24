import torch
import torch.nn as nn
from torch_scatter import scatter


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relation, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.num_relation = num_relation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act

    def forward(self, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
