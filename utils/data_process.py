import torch
import random


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
                             num_entity):
    nagative = data.clone()
    rate = torch.rand(nagative.shape[0])
    mask = rate < 0.5
    nagative[:, 0][mask] = (nagative[:, 0][mask] + torch.randint(1, num_entity,
                                                                 (nagative[:, 0][mask].shape[0],),
                                                                 device=nagative.device)) % num_entity
    mask = rate >= 0.5
    nagative[:, 2][mask] = (nagative[:, 2][mask] + torch.randint(1, num_entity,
                                                                 (nagative[:, 0][mask].shape[0],),
                                                                 device=nagative.device)) % num_entity
    return nagative


def batch_data(data: torch.Tensor,
               batch_size=256,
               shuffle=True,
               label=None):
    size = len(data)
    num_batch = int(size / batch_size) + int(size % batch_size != 0)
    index = list(range(size))
    random.shuffle(index)
    for i in range(num_batch):
        if batch_size * i + batch_size <= size:
            b_index = index[batch_size * i:batch_size * i + batch_size]
        else:
            b_index = index[batch_size * i:size]
        if label is None:
            yield data[b_index]
        else:
            yield data[b_index], label[b_index]


def float_to_int_exp(num) -> str:
    exp = 0
    while num % 1 != 0:
        num *= 10
        exp += 1
    if exp == 0:
        return str(int(num))
    return str(int(num)) + 'e-' + str(exp)


def add_reverse_relation(edges: list,
                         num_relation):
    res = []
    for edge in edges:
        reverse = torch.cat([edge[:, 2].unsqueeze(1),
                             edge[:, 1].unsqueeze(1) + num_relation,
                             edge[:, 0].unsqueeze(1)],
                            dim=1)
        res.append(torch.cat([edge, reverse], dim=0))
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
    ans = torch.index_select(ans, dim=0, index=i).to_dense()
    ans[range(len(ans)), data[:, 2]] = 0
    return score + ans * -10000
