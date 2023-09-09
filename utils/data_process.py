import torch
from collections import deque


def reverse_dict(dict_t: dict):
    new_dict = {}
    for key in dict_t.keys():
        new_dict[dict_t[key]] = key
    return new_dict


def split_data_by_time(data: torch.Tensor,
                       start=0):
    data_split = []
    time_index = {}
    times, _ = torch.unique(data[:, 3]).sort()
    for i in times:
        time_index[i.item()] = len(data_split)
        data_split.append(data[data[:, 3] == i][:, 0:3])
    return data_split, time_index, times


def generate_negative_sample(data: torch.Tensor,
                             num_entity: int,
                             mode='random',
                             total_positive=None,
                             index=None):
    nagative = data.clone()
    rate = torch.rand(nagative.shape[0])
    if mode == 'random':
        mask = rate < 0.5
        nagative[:, 0][mask] = (nagative[:, 0][mask] + torch.randint(1, num_entity,
                                                                     (nagative[:, 0][mask].shape[0],),
                                                                     device=nagative.device)) % num_entity
        mask = rate >= 0.5
        nagative[:, 2][mask] = (nagative[:, 2][mask] + torch.randint(1, num_entity,
                                                                     (nagative[:, 0][mask].shape[0],),
                                                                     device=nagative.device)) % num_entity
    elif mode == 'strict':
        """batch_ans = total_positive.index_select(index).todense()
        zero_indices = torch.nonzero(batch_ans == 0)
        x, i, c = torch.unique(zero_indices[:, 0], return_counts=True, return_inverse=True)
        abs_index = torch.nonzero(zero_indices[:, 0][1:] != zero_indices[:, 0][:-1])[:, 0] + 1
        indices = torch.cat([torch.LongTensor([0]), abs_index])
        rlt_index = torch.randint(num_entity, size=c.shape) % c
        full_index = abs_index + rlt_index
        sample = zero_indices[full_index]"""
        raise NotImplementedError
    return nagative


def batch_data(data: torch.Tensor,
               batch_size=256,
               shuffle=True,
               label=None):
    size = len(data)
    num_batch = int(size / batch_size) + int(size % batch_size != 0)
    index = torch.randperm(size)
    for i in range(num_batch):
        if batch_size * i + batch_size <= size:
            b_index = index[batch_size * i:batch_size * i + batch_size]
        else:
            b_index = index[batch_size * i:size]
        if label is None:
            yield data[b_index]
        else:
            yield data[b_index], label[b_index]


def add_inverse(edge: torch.Tensor,
                num_relation):
    reverse = torch.cat([edge[:, 2].unsqueeze(1),
                         edge[:, 1].unsqueeze(1) + num_relation,
                         edge[:, 0].unsqueeze(1)],
                        dim=1)
    return torch.cat([edge, reverse], dim=0)


def add_reverse_relation(edges: list,
                         num_relation):
    res = []
    for edge in edges:
        res.append(add_inverse(edge, num_relation))
    return res


def get_answer(data, num_entity, num_relation):
    i = data[:, 0] * num_relation + data[:, 1]
    i = torch.cat([i.unsqueeze(0), data[:, 2].unsqueeze(0)], dim=0)
    v = torch.ones(i.shape[-1])
    ans = torch.sparse_coo_tensor(i, v, size=(num_entity * num_relation, num_entity), device=data.device)
    return ans


def filter_score(score: torch.Tensor,
                 ans: torch.Tensor,
                 data: torch.Tensor,
                 num_relation):
    i = data[:, 0] * num_relation + data[:, 1]
    max_score = torch.max(score)
    mask = torch.index_select(ans, dim=0, index=i).to_dense()
    mask[range(len(mask)), data[:, 2]] = 0
    return score - mask * max_score


def load_data(file: str,
              load_time=False,
              encoding='utf-8'):
    data = []
    with open(file, encoding=encoding) as f:
        content = f.read()
        content = content.strip()
        content = content.split("\n")
        for line in content:
            fact = line.split()
            if load_time:
                data.append([int(fact[0]), int(fact[1]), int(fact[2]), int(fact[3])])
            else:
                data.append([int(fact[0]), int(fact[1]), int(fact[2])])
    data = torch.LongTensor(data)
    return data


def load_dict(file: str,
              encoding='utf-8'):
    dict_data = {}
    with open(file, encoding=encoding) as f:
        content = f.read()
        content = content.strip()
        content = content.split("\n")
        for line in content:
            items = line.split('\t')
            dict_data[items[0]] = int(items[1])
    return dict_data


def bi_bfs(ad_matrix, start, end):
    """"
    Bidirectional Breadth First Search
    :param end: destination node
    :param start: source node
    :param ad_matrix: adjacency matrix
    """
    if start == end:
        return [start]

    def get_neighbors(v, dim=0):
        """
        get neighbors and relations
        :param v:
        :param dim:
        :return:
        """
        index = torch.index_select(ad_matrix, index=torch.LongTensor([v, ]), dim=dim).to_dense()
        x = torch.nonzero(index)
        if dim == 0:
            neighbors = torch.nonzero(index)[:, 1]
        else:
            neighbors = torch.nonzero(index)[:, 0]
        rela_id = index.squeeze()[neighbors] - 1
        return zip(neighbors, rela_id)

    start_queue, end_queue = deque(), deque()
    start_queue.append(start)
    end_queue.append(end)

    # visited record, the value of which is the previous node and the relation which is between current node
    # and previous node.the key is current node.
    s_visited = {start: (-1, -1)}
    e_visited = {end: (-1, -1)}

    def generate_path(last_node, visited, reverse=False, take_last_node=False):
        """
        generate path from the visited record
        :param last_node:
        :param visited:
        :param reverse:
        :param take_last_node:
        :return:
        """
        path = []
        if take_last_node:
            path.append(last_node)
        rela_id, father = visited[last_node]
        while father != -1:
            path.append(rela_id)
            path.append(father)
            rela_id, father = visited[father]
        if reverse:
            path = path[::-1]
        return path

    while len(start_queue) and len(end_queue):
        start_size, end_size = len(start_queue), len(end_queue)

        for _ in range(start_size):
            cur = start_queue.popleft()
            for item in get_neighbors(cur):
                neighbor, rela = item
                neighbor, rela = neighbor.item(), rela.item()
                if neighbor in s_visited.keys():
                    continue
                elif neighbor in e_visited.keys():
                    s_visited[neighbor] = (rela, cur)
                    return generate_path(neighbor, s_visited, reverse=True, take_last_node=False) \
                           + generate_path(neighbor, e_visited, reverse=False, take_last_node=True)

                else:
                    s_visited[neighbor] = (rela, cur)
                    start_queue.append(neighbor)

        for _ in range(end_size):
            cur = end_queue.popleft()
            for item in get_neighbors(cur, dim=1):
                neighbor, rela = item
                neighbor, rela = neighbor.item(), rela.item()
                if neighbor in e_visited.keys():
                    continue
                elif neighbor in s_visited.keys():
                    e_visited[neighbor] = (rela, cur)
                    return generate_path(neighbor, s_visited, reverse=True, take_last_node=True) \
                           + generate_path(neighbor, e_visited, reverse=False, take_last_node=False)
                else:
                    e_visited[neighbor] = (rela, cur)
                    end_queue.append(neighbor)
    return


def generate_shortest_path(triplets: torch.Tensor,
                           num_entity: int,
                           source: torch.Tensor,
                           destination: torch.Tensor):
    """
        find the shortest path from source node to destination node on graph which is constructed by triplets.
        :param triplets:
        :param num_entity:
        :param source:
        :param destination:
        :return: a dictionary , the key and value of which is (source, destination) and the path between source and
        destination
        """
    i = torch.cat([triplets[:, 0].unsqueeze(0), triplets[:, 2].unsqueeze(0)], dim=0)
    ad_matrix = torch.sparse_coo_tensor(i, triplets[:, 1] + 1, size=(num_entity, num_entity),
                                        device=triplets.device).coalesce()
    paths = {}
    for s, d in zip(source, destination):
        path = bi_bfs(ad_matrix, s.item(), d.item())
        paths[(s.item(), d.item())] = path
    return paths


def label_smooth(label, num_class, epsilon):
    """

    :param num_class:
    :param label:label index
    :return:
    """
    y = torch.zeros((label.shape[0], num_class), device=label.device)
    y.scatter_(1, label.unsqueeze(1), 1)
    y = (1.0 - epsilon) * y + (1.0 / num_class)
    return y
