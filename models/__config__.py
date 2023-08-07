from models.cygnet import CyGNet
from models.regcn import REGCN
from models.cen import CEN
from models.cenet import CeNet
model_list = {
    'cygnet': CyGNet,
    'regcn': REGCN,
    'cen': CEN,
    'cenet':CeNet
}


def get_default_model(model: str):
    return model
