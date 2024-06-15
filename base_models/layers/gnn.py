import torch


class GNN:
    @staticmethod
    def gcn_aggregate(message,
                      des,
                      normalize='mean'):
        """
        aggregate messages which are sent to the same destination node
        :param message:
        :param des: destination for each message
        :param normalize: weather normalize the massage aggregated
        :return:
        """
        des_unique, des_index, count = torch.unique(des, return_inverse=True, return_counts=True)
        message = torch.zeros(des_unique.shape[0],
                              message.shape[1],
                              dtype=message.dtype,
                              device=message.device).scatter_add_(0,
                                                                  des_index.unsqueeze(1).expand_as(message),
                                                                  message)
        if normalize is not None:
            if normalize == 'mean':
                message = message / count.reshape(-1, 1)
        return des_unique, message

    @staticmethod
    def gcn_aggregate_adj(message,
                          adj):
        """
        use adjacency matrix to aggregate message
        :param message:
        :param adj: torch.sparse_coo_tensor, adjacency matrix or modified adjacency matrix
        :return:
        """
        return torch.sparse.mm(adj, message)

    @staticmethod
    def edges2adj(edges, num_entity):
        src, rela, dst = edges.transpose(0, 1)
        i = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        v = torch.ones(edges.shape[0], device=edges.device)
        adj = torch.sparse_coo_tensor(i, v, size=(num_entity, num_entity), device=edges.device)
        return adj

    @staticmethod
    def cal_out_degree(adj):
        out_dgr = torch.sparse.sum(adj, dim=-1)
        return out_dgr
