import torch
import json
import os


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


def save_json(content,
            path: str,
            name: str):
    """
    save experimental data
    :param content: content to save
    :param path: saved file path
    :param name: saved file name
    :return: None
    """
    if not os.path.exists(path):
        # create new directory
        os.mkdir(path)
    f = open(path + name + '.json', 'w', encoding='utf-8')
    json.dump(content, f)
    f.close()


def load_json(path: str, name: str):
    """
    load data
    :param path:file path
    :param name: file name
    :return: data loaded
    """
    f = open(path + name + '.json', 'r', encoding='utf-8')
    content = json.load(f)
    f.close()
    return content
