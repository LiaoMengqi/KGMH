import torch
from base_models.regcn_base import REGCNBase
from base_models.cygnet_base import CyGNetBase
from base_models.cen_base import CENBase

from data.data_loader import DataLoader

base_models_list = {
    'regcn': REGCNBase,
    'cygnet': CyGNetBase,
    'cen': CENBase
}


def get_base_model(args,
                   data: DataLoader,
                   ) -> torch.nn.Module:
    """
    get base model with configuration
    :param data: dataset
    :param args: parameters
    :return: base model
    """
    raise NotImplementedError


def get_default_base_model(model: str,
                           data: DataLoader) -> torch.nn.Module:
    """
    get base model with default parameter
    :param model: model name
    :param data: dataset
    :return: base model
    """
    if model == 'regcn':
        base_model = REGCNBase(
            num_entity=data.num_entity,
            num_relation=data.num_relation,
            hidden_dim=50,
            seq_len=3,
            num_layer=2,
            dropout=0.2,
            active=True,
            self_loop=True,
            layer_norm=True)
    elif model == 'cygnet':
        base_model = CyGNetBase(
            num_entity=data.num_entity,
            num_relation=data.num_relation,
            h_dim=200,
            alpha=0.8,
            penalty=-100
        )
    elif model == 'cen':
        base_model = CENBase(
            num_entity=data.num_entity,
            num_relation=data.num_relation,
            dim=50,
            dropout=0.2,
            seq_len=3,
            channel=50,
            width=3
        )
    else:
        raise Exception('model ' + model + ' not exist!')
    return base_model
