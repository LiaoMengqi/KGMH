import torch
from base_models.regcn_base import REGCNBase
from data.data_loader import DataLoader

base_models_list = {
    'regcn': REGCNBase
}


def get_default_base_model(model: str, data: DataLoader) -> torch.nn.Module:
    base_model = None
    if model == 'regcn':
        base_model = REGCNBase(
            num_entity=data.num_entity,
            num_relation=data.num_relation,
            hidden_dim=64,
            seq_len=10,
            num_layer=2,
            dropout=0.2,
            active=True,
            self_loop=True,
            layer_norm=True)
    else:
        raise Exception('model ' + model + ' not exist!')
    return base_model
