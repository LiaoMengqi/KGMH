import argparse
from models.__config__ import *
from base_models.__config__ import *
import torch
from utils.io_func import save_checkpoint
from utils.io_func import load_checkpoint
from utils.io_func import to_json
from utils.plot import hist_value
import time

opt_list = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}


def get_opt(args,
            model: torch.nn.Module):
    opt = None
    if args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad,
                               eps=args.eps)

    return opt


def train(model, epochs, batch_size, step, early_stop):
    """
    train model
    :param early_stop:
    :param model: model
    :param epochs: train epoch
    :param batch_size: batch size
    :param step: step to evaluate model on valid set
    :return: None
    """
    name = model.get_name()
    metric_history = {}
    loss_history = []
    train_time = []
    evaluate_time = []
    decline = 0
    best = 0
    for epoch in range(epochs):
        time_start = time.time()
        loss = model.train_epoch(batch_size=batch_size)
        time_end = time.time()
        train_time.append(time_end - time_start)
        loss_history.append(loss)
        print('epoch: %d |loss: %f |time: %fs' % (epoch + 1, loss, time_end - time_start))

        if (epoch + 1) % step == 0:
            time_start = time.time()
            metrics = model.test(batch_size=batch_size)
            time_end = time.time()
            evaluate_time.append(time_end - time_start)
            print('hits@1: %f |hits@3: %f |hits@10: %f |hits@100: %f |mr: %f |mrr: %f |time: %f' %
                  (metrics['hits@1'],
                   metrics['hits@3'],
                   metrics['hits@10'],
                   metrics['hits@100'],
                   metrics['mr'],
                   metrics['mrr'],
                   time_end - time_start))
            for key in metrics.keys():
                if key not in metric_history.keys():
                    metric_history[key] = []
                    metric_history[key].append(metrics[key])
                else:
                    metric_history[key].append(metrics[key])
            if early_stop > 0 and epoch > 0:
                if metric_history['mrr'][-1] < metric_history['mrr'][-2]:
                    decline = decline + 1
                if decline >= early_stop:
                    break
            if metrics['mrr'] > metric_history['mrr'][best]:
                best = epoch
    print("\n**********************************finish**********************************\n")
    print("best : hits@1: %f |hits@3: %f |hits@10: %f |hits@100: %f |mr: %f |mrr: %f" %
          (metric_history['hits@1'][best],
           metric_history['hits@3'][best],
           metric_history['hits@10'][best],
           metric_history['hits@100'][best],
           metric_history['mr'][best],
           metric_history['mrr'][best],))

    # plot
    hist_value({'hits@1': metric_history['hits@1'],
                'hits@3': metric_history['hits@3'],
                'hits@10': metric_history['hits@10'],
                'hits@100': metric_history['hits@100']},
               value='hits@k',
               name=name + '_valid_hits@k')
    hist_value({'mr': metric_history['mr']},
               value='mr',
               name=name + '_valid_mr')
    hist_value({'mrr': metric_history['mrr']},
               value='mrr',
               name=name + '_valid_mrr')
    hist_value({'loss': loss_history},
               value='loss',
               name=name + '_valid_loss')
    # save model
    save_checkpoint(model, name=name)
    # save train history
    data_to_save = metric_history
    data_to_save['loss'] = loss_history
    data_to_save['train_time'] = train_time
    data_to_save['evaluate_time'] = evaluate_time
    to_json(data_to_save, name=name)


def evaluate(model, batch_size, data='test'):
    """
    evaluate model in test set or valid set
    :param model: model
    :param batch_size: batch size
    :param data: dataset
    :return: None
    """
    name = model.get_name()
    metrics = model.test(batch_size=batch_size, dataset='test')
    print('hits@1 %f |hits@3 %f |hits@10 %f |hits@100 %f |mr %f |mrr %f' % (metrics['hits@1'],
                                                                            metrics['hits@3'],
                                                                            metrics['hits@10'],
                                                                            metrics['hits@100'],
                                                                            metrics['mr'],
                                                                            metrics['mrr']))
    to_json(metrics, name=name + '_test_result')


def main(args):
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
        base_model = get_base_model(args, data)
    else:
        base_model = get_default_base_model(args.model, data)
    base_model.to(device)
    # Optimizer
    opt = get_opt(args, base_model)
    # model
    model = model_list[args.model](base_model, data, opt)
    model.to(device)
    # load checkpoint
    if args.checkpoint is not None:
        load_checkpoint(model, name=args.checkpoint)
    if args.test:
        # evaluate
        if args.checkpoint is None:
            raise Exception("You need to load a checkpoint for testing!")
        evaluate(model, args.batch_size)
    else:
        # train
        train(model, args.epoch, args.batch_size, args.eva_step, args.early_stop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGTL')
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="choose model")
    parser.add_argument("--conf", action='store_true', default=False,
                        help="configure parameter")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path and name of model saved')
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
                        help="optimizer parameter")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="optimizer parameter")
    parser.add_argument("--amsgrad", action='store_true', default=False,
                        help="Adam optimizer parameter")

    # train
    parser.add_argument("--epoch", type=int, default=15,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size.")
    parser.add_argument("--eva-step", type=int, default=1,
                        help="evaluate model on valid set after 'eva-step' step of training.")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="patience for early stop.")
    # test
    parser.add_argument("--test", action='store_true', default=False,
                        help="evaluate model on test set, and notice that you must load a checkpoint for this.")
    # other
    parser.add_argument("--gpu", action='store_true', default=True,
                        help="use GPU.")
    args_parsed = parser.parse_args()

    main(args_parsed)
