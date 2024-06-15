import torch
from base_models.layers.gnn import GNN


class GATLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, head_num, leakey_rate=1e-2):
        super(GATLayer, self).__init__()
        self.head_num = head_num

        # initial weight

        W = torch.rand(size=(head_num, output_dim, input_dim))
        torch.nn.init.xavier_normal_(W)

        a = torch.rand(size=(head_num, 1, output_dim * 2))
        gain = torch.nn.init.calculate_gain('leaky_relu', leakey_rate)
        torch.nn.init.xavier_normal_(a, gain)

        self.W = torch.nn.Parameter(W)
        self.a = torch.nn.Parameter(a)

        self.output_dim = output_dim
        self.leakey_relu = torch.nn.LeakyReLU(leakey_rate)

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

        x = torch.matmul(self.W, input_h.T).transpose(1, 2)
        h = x.sum(dim=0)/self.head_num

        # message
        a = torch.softmax(
            self.leakey_relu((torch.cat([x[:, src], x[:, dst]], dim=-1) * self.a).sum(dim=-1)),
            dim=-1
        ).unsqueeze(-1)
        message = (x[:, dst] * a).reshape(-1, self.output_dim)

        # aggregate
        src = torch.cat([src] * self.head_num)
        i, message = GNN.gcn_aggregate(message, src, normalize=None)

        # h = torch.zeros(size=(x.shape[1], x.shape[2]), device=x.device)
        h[i] = h[i] + message / self.head_num

        return h
