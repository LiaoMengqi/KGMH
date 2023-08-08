import torch


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
