from models.cygnet import CyGNet
from models.regcn import REGCN

model_list = {
    'cygnet': CyGNet,
    'regcn': REGCN
}


def get_default_model(model: str):
    return model
