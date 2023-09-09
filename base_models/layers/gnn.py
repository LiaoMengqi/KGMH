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
