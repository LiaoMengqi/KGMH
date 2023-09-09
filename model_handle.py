from utils.func import *
from data.data_loader import DataLoader


class ModelHandle:
    def __init__(self, model):
        self.model = model
        self.BaseModel, self.Model = self.import_model(model)

    @staticmethod
    def get_type(model_name):
        if model_name in ['transe', 'rgcn', 'sacn', 'distmult', 'gcn']:
            return 'static'
        elif model_name in ['cygnet', 'regcn', 'cen', 'cenet']:
            return 'temporal/extrapolation'
        else:
            raise Exception

    def import_model(self, model):
        # temporal/extrapolation
        if model == 'cygnet':
            from base_models.cygnet_base import CyGNetBase as BaseModel
            from models.cygnet import CyGNet as Model
        elif model == 'regcn':
            from base_models.regcn_base import REGCNBase as BaseModel
            from models.regcn import REGCN as Model
        elif model == 'cen':
            from base_models.cen_base import CENBase as BaseModel
            from models.cen import CEN as Model
        elif model == 'cenet':
            from base_models.cenet_base import CeNetBase as BaseModel
            from models.cenet import CeNet as Model
        # static
        elif model == 'transe':
            from base_models.transe_base import TransEBase as BaseModel
            from models.transe import TransE as Model
        elif model == 'rgcn':
            from base_models.rgcn_base import RGCNBase as BaseModel
            from models.rgcn import RGCN as Model
        elif model == 'distmult':
            from base_models.distmult_base import DistMultBase as BaseModel
            from models.distmult import DistMult as Model
        elif model == 'sacn':
            from base_models.sacn_base import SACNBase as BaseModel
            from models.sacn import SACN as Model
        else:
            raise Exception('Model error!')
        return BaseModel, Model

    def get_base_model(self,
                       data: DataLoader, ):
        config = {}
        config['num_entity'] = data.num_entity
        config['num_relation'] = data.num_relation
        # temporal/extrapolation
        if self.model == 'cygnet':
            hints = [
                ["input hidden dimension (int, >0):", int, 'h_dim'],
                ["input alpha (float, [0,1]):", float, 'alpha'],
                ["input penalty (float, <0):", float, 'penalty']
            ]
        elif self.model == 'regcn':
            hints = [
                ["input hidden dimension (int, >0):", int, 'hidden_dim'],
                ["input sequence length (int, >0):", int, 'seq_len'],
                ["input The number of graph convolutional layers (int, >0):", int, 'num_layer'],
                ["input dropout probability (float ,[0,1)):", float, "dropout"],
                ["whether to use active function (bool, 0 or 1):", bool, "active"],
                ["whether to use self loop (bool, 0 or 1):", bool, "self_loop"],
                ["whether to use normalization (bool, 0 or 1):", bool, "layer_norm"]
            ]
        elif self.model == 'cen':
            hints = [
                ["input hidden dimension (int, >0):", int, 'dim'],
                ["input dropout probability (float ,[0,1)):", float, "dropout"],
                ["input the number of convolution channels (int, >0):", int, 'channel'],
                ["input the length of convolution kernel (int, >0):", int, 'width'],
                ["input the length of history sequence to learn (int ,>0):", int, 'seq_len'],
                ['whether to use layer normalization (bool, 0 or 1):', bool, 'layer_norm']
            ]
        elif self.model == 'cenet':
            hints = [
                ["input hidden dimension (int, >0):", int, 'dim'],
                ["input dropout probability (float ,[0,1)):", float, "drop_prop"],
                ["input lambda (float, >0):", float, 'lambdax'],
                ['input alpha (float, (0,1]):', float, 'alpha'],
                ['input witch mask mode to use (str, \'soft\' or \'hard\')', str, 'mode']
            ]
        # static
        elif self.model == 'transe':
            hints = [
                ["input embedding dimension (int, >0):", int, 'emb_dim'],
                ["input margin (float, >0):", float, "margin"],
                ["input normalization p (float, >0):", float, "p_norm"],
                ['input coefficient of entity scale loss (float, >0):', float, 'c_e'],
                ['input coefficient of relation scale loss (float, >0):', float, 'c_r']
            ]
        elif self.model == 'rgcn':
            hints = [
                ["input dimension list of R-GCN layer (list[int], number of item >=2, value of item >0):", list,
                 'dim_list', int],
                ["whether use basis decompose (bool, 0 or 1):", bool, "use_basis"],
                ["if use basis decompose, input the number of basis (int, >0):", int, "num_basis"],
                ["if don't use basis decompose, whether use block decompose (bool, 0 or 1):", bool, "use_block"],
                [
                    "if use block decompose, input the size of block, "
                    "each dimension must can be divided by the size of block (int, >0):",
                    int, "num_block"],
                ["input dropout probability of self-loop (float ,[0,1)):", float, "dropout_s"],
                ["input dropout probability of non-self-loop (float ,[0,1)):", float, "dropout_o"]
            ]
        elif self.model == 'distmult':
            hints = [
                ["input the input dimension (int,  >0):", int, 'input_dim'],
                ["input the output dimension (int,  >0):", int, 'output_dim']
            ]
        elif self.model == 'sacn':
            hints = [
                ["input the input dimension (int,  >0):", int, 'input_dim'],
                ["input the number of WGCN layers (int,  >0):", int, 'num_layer'],
                ["input the number of channels (int, >0):", int, "num_channel"],
                ["input the length of convolution kernel (int, >0)", int, "kernel_length"],
                ["input dropout probability (float ,[0,1)):", float, "dropout"]
            ]
        else:
            raise Exception
        for hint in hints:
            flag = True
            while flag:
                try:
                    input_text = input(hint[0])
                    if hint[1] == bool:
                        config[hint[2]] = hint[1](int(input_text))
                    elif hint[1] == list:
                        input_text = input_text.replace('[', '')
                        input_text = input_text.replace(']', '')
                        input_text = input_text.split(',')
                        for i in range(len(input_text)):
                            input_text[i] = hint[3](input_text[i])
                        config[hint[2]] = input_text
                    else:
                        config[hint[2]] = hint[1](input_text)
                except ValueError:
                    print('input error!')
                else:
                    flag = False
        return self.get_base_model_from_config(config=config)

    def get_base_model_from_config(self, config=None, data=None):
        default = False
        if config is None:
            default = True

        if self.model == 'cygnet':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                h_dim=50 if default else config['h_dim'],
                alpha=0.5 if default else config['alpha'],
                penalty=1 if default else config['penalty']
            )
        elif self.model == 'regcn':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                hidden_dim=50 if default else config['hidden_dim'],
                seq_len=10 if default else config['seq_len'],
                num_layer=3 if default else config['num_layer'],
                dropout=0.4 if default else config['dropout'],
                active=True if default else config['active'],
                self_loop=True if default else config['self_loop'],
                layer_norm=True if default else config['layer_norm'],
            )
        elif self.model == 'cen':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                dim=50 if default else config['dim'],
                dropout=0.4 if default else config['dropout'],
                channel=50 if default else config['channel'],
                width=3 if default else config['width'],
                seq_len=3 if default else config['seq_len'],
                layer_norm=True if default else config['layer_norm']
            )
        elif self.model == 'cenet':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                dim=200 if default else config['dim'],
                drop_prop=0.4 if default else config['drop_prop'],
                lambdax=2.0 if default else config['lambdax'],
                alpha=0.2 if default else config['alpha'],
                mode='soft' if default else config['mode'],
            )
        # static
        elif self.model == 'transe':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                emb_dim=200 if default else config['emb_dim'],
                margin=2.0 if default else config['margin'],
                p_norm=1 if default else config['p_norm'],
                c_e=0 if default else config['c_e'],
                c_r=0 if default else config['c_r']
            )
        elif self.model == 'rgcn':
            base_model = self.BaseModel(
                dims=[64, 64],
                num_relation=data.num_relation if default else config['num_relation'],
                num_entity=data.num_entity if default else config['num_entity'],
                use_basis=True if default else config['use_basis'],
                num_basis=2 if default else config['num_basis'],
                use_block=False if default else config['use_block'],
                num_block=10 if default else config['num_block'],
                dropout_s=0 if default else config['dropout_s'],
                dropout_o=0 if default else config['dropout_o'],
                inverse=False if default else config['inverse']
            )
        elif self.model == 'distmult':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                input_dim=50 if default else config['input_dim'],
                output_dim=50 if default else config['output_dim']
            )
        elif self.model == 'sacn':
            base_model = self.BaseModel(
                num_entity=data.num_entity if default else config['num_entity'],
                num_relation=data.num_relation if default else config['num_relation'],
                dim=200 if default else config['dim'],
                num_layer=2 if default else config['num_layer'],
                num_channel=100 if default else config['num_channel'],
                kernel_length=3 if default else config['kernel_length'],
                dropout=0.2 if default else config['dropout'],
            )
        else:
            raise Exception('model not exist!')
        return base_model

    def get_default_base_model(self,
                               data: DataLoader) -> torch.nn.Module:
        """
        get base model with default parameter
        :param data: dataset
        :return: base model
        """
        base_model = self.get_base_model_from_config(config=None, data=data)
        return base_model


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
    config = model.get_config()
    full_path = path + config['model'] + '/' + name + '/'
    if not os.path.exists(full_path):
        # create new directory
        os.makedirs(full_path)
    save_json(config, path=full_path, name='config')
    torch.save(model.state_dict(), full_path + 'model_params.pth')


def load_checkpoint(checkpoint: str,
                    model_handle: ModelHandle,
                    args,
                    device,
                    path='./checkpoint/'):
    """
    load checkpoint
    :param device:
    :param args:
    :param path: path where checkpoint saved
    :return: None
    """
    path = path + args.model + '/' + checkpoint + '/'
    if not os.path.exists(path):
        # create new directory
        raise Exception('The path \'' + path + '\' don\'t exist!')
    config = load_json(path=path, name='config')
    base_model = model_handle.get_base_model_from_config(config)
    data = DataLoader(config['dataset'], root_path='./data/', type=model_handle.get_type(model_handle.model))
    data.load()
    data.to(device)
    opt = get_optimizer(args, base_model)
    model = model_handle.Model(base_model, data, opt)
    model.load_state_dict(torch.load(path + 'model_params.pth'))
    model.to(device)
    return model
