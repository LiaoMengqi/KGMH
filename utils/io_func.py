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
            items = line.split()
            dict_data[items[0]] = int(items[1])
    return dict_data


def to_json(content,
            path='./result/',
            name='model'):
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


def from_json(path='./result/',
              name='model'):
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


def save_checkpoint(model: torch.nn.Module,
                    name: str,
                    path='./checkpoint/'):
    """
    save checkpoint
    :param model: model
    :param name: model name
    :param path: path to save
    :return: None
    """
    if not os.path.exists(path):
        # create new directory
        os.mkdir(path)
    torch.save(model.state_dict(), path + name)


def load_checkpoint(model: torch.nn.Module,
                    name: str,
                    path='./checkpoint/'):
    """
    load checkpoint
    :param model: model
    :param name: model name
    :param path: path where checkpoint saved
    :return: None
    """
    if not os.path.exists(path + name):
        # create new directory
        raise Exception('There is no checkpoint named ' + name + ' in ' + path)
    model.load_state_dict(torch.load(path + name))
