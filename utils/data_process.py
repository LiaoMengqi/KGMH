import numpy as np


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
    data = np.array(data, dtype=np.int64)
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


def split_data_by_time(data: np.array):
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
        data_split[i] = np.array(data_split[i])
    return data_split, time_index, time_list
