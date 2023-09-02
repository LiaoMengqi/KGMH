import torch
import json
import os
import random
import numpy as np
import subprocess
import re


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
        os.makedirs(path)
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


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_default_fp(fp: str):
    if fp == 'fp16':
        torch.set_default_dtype(torch.float16)
    elif fp == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    elif fp == 'fp64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)


def select_gpu():
    try:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    return sorted_used[0][0]


def set_device(gpu):
    if gpu != -1:
        # use gpu
        if not torch.cuda.is_available():
            # gpu not available
            print('No GPU available. Using CPU.')
            device = 'cpu'
        else:
            # gpu available
            if gpu < -1:
                # auto select gpu
                gpu_id = select_gpu()
                print('Auto select gpu:%d' % gpu_id)
                device = 'cuda:%d' % gpu_id
            else:
                # specify gpu id
                if gpu >= torch.cuda.device_count():
                    gpu_id = select_gpu()
                    print('GPU id is invalid. Auto select gpu:%d' % gpu_id)
                    device = 'cuda:%d' % gpu_id
                else:
                    print('Using gpu:%d' % gpu)
                    device = 'cuda:%d' % gpu
    else:
        print('Using CPU.')
        device = 'cpu'
    return device


def get_optimizer(args,
                  model: torch.nn.Module):
    opt = None
    if args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad,
                               eps=args.eps)
    return opt
