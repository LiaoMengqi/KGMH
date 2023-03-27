import torch


def load_data(file: str, load_time=False, encoding='utf-8'):
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


def load_dict(file: str, encoding='utf-8'):
    dict_data = {}
    with open(file, encoding=encoding) as f:
        content = f.read()
        content = content.strip()
        content = content.split("\n")
        for line in content:
            items = line.split()
            dict_data[items[0]] = int(items[1])
    return dict_data


def reverse_dict(dict_t: dict):
    new_dict = {}
    for key in dict_t.keys():
        new_dict[dict_t[key]] = key
    return new_dict


def split_data_by_time(data: torch.Tensor):
    data_split = []
    time_index = {}
    time_list = []
    for line in data:
        if line[3] in time_list:
            data_split[time_index[line[3]]].append(line[0:3])
        else:
            time_index[line[3]] = len(data_split)
            time_list.append(line[3])
            data_split.append([line[0:3]])
    for i in range(len(data_split)):
        data_split[i] = torch.LongTensor(data_split[i])
    return data_split, time_index, time_list


def generate_negative_sample(data: torch.Tensor, num_entity):
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


def batch_data(data: torch.Tensor, batch_size):
    data_shuffled = data[torch.randperm(data.shape[0])]
    batch_num = int(data.shape[0] / batch_size) + int(data.shape[0] % batch_size != 0)
    res = []
    for i in range(batch_num):
        if i * batch_size + batch_size < data.shape[0]:
            res.append(data_shuffled[i * batch_size:i * batch_size + batch_size])
        else:
            res.append(data_shuffled[i * batch_size:data.shape[0]])
    return res


def float_to_int_exp(num):
    exp = 0
    while num % 1 != 0:
        num *= 10
        exp += 1
    return int(num), exp
