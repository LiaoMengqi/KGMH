import argparse
from models.__list__ import *
from base_models.__list__ import *
import torch

opt_list = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}


def run(args):
    # choose device
    if args.gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    # load data
    data = DataLoader(args.dataset, './data/temporal/extrapolation')
    data.load(load_time=True)
    data.to(device)
    # base model
    if args.conf:
        base_model = None
    else:
        base_model = get_default_base_model(args.model, data)
    base_model.to(device)
    # Optimizer
    opt = opt_list[args.opt](base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model
    model = model_list[args.model](base_model, data, opt)
    model.to(device)
    for epoch in range(args.epoch):
        loss = model.train_epoch()
        print('epoch: %d |loss: %f ' % (epoch + 1, loss))
        if (epoch + 1) % args.eva_step == 0:
            metrics = model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGTL')
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="choose model")
    parser.add_argument("--conf", action='store_true', default=False,
                        help="configure parameter")
    # dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="choose dataset")
    # Optimizer
    parser.add_argument("--opt", type=str, default='adam',
                        help="optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="weight-decay")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="momentum")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    # train
    parser.add_argument("--epoch", type=int, default=30,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="learning rate")
    parser.add_argument("--eva-step", type=int, default=5,
                        help="learning rate")
    # other
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--gpu", action='store_true', default=True,
                        help="use GPU")
    args_parsed = parser.parse_args()
    run(args_parsed)
