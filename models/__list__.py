from models.cygnet import CyGNet
from models.regcn import REGCN
from models.cen import CEN

model_list = {
    'cygnet': CyGNet,
    'regcn': REGCN,
    'cen': CEN
}


def get_default_model(model: str):
    return model
