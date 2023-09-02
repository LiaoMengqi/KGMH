import argparse
import time

from model_handle import *


def train(model, epochs, batch_size, step, early_stop, monitor, filter_out=False, plot=False):
    """
    train model
    :param monitor:
    :param plot:
    :param filter_out:
    :param early_stop:
    :param model: model
    :param epochs: train epoch
    :param batch_size: batch size
    :param step: step to evaluate model on valid set
    :return: None
    """
    model_id = time.strftime('%Y%m%d%H%M%S', time.localtime())
    metric_history = {}
    loss_history = []
    train_time = []
    evaluate_time = []
    if filter_out:
        monitor = 'filter ' + monitor
    best = 0
    tolerance = early_stop
    for epoch in range(epochs):
        time_start = time.time()
        loss = model.train_epoch(batch_size=batch_size)
        time_end = time.time()
        train_time.append(time_end - time_start)
        loss_history.append(float(loss))
        print('epoch: %d |loss: %f |time: %fs' % (epoch + 1, loss, time_end - time_start))
        if (epoch + 1) % step == 0:
            time_start = time.time()
            metrics = model.test(batch_size=batch_size, filter_out=filter_out)
            time_end = time.time()
            evaluate_time.append(time_end - time_start)

            for key in sorted(metrics.keys()):
                print(key, ': ', metrics[key], ' |', end='')
                if key not in metric_history.keys():
                    metric_history[key] = []
                    metric_history[key].append(metrics[key])
                else:
                    metric_history[key].append(metrics[key])
            print('time: %f' % (time_end - time_start))
            if metric_history[monitor][-1] < metric_history[monitor][best]:
                # performance decline
                if early_stop > 0:
                    tolerance -= 1
                    if tolerance <= 0:
                        break
            else:
                # achieve better performance, save model
                save_checkpoint(model, name=model_id)
                # reset tolerance
                tolerance = early_stop
                best = (epoch // step)

    print("\n**********************************finish**********************************\n")
    print("best : ", end='')
    for key in sorted(metric_history.keys()):
        print(key, ' ', metric_history[key][best], ' |', end='')
    print()
    path = './checkpoint/' + model.name + '/' + model_id + '/'
    if plot:
        # plot loss and metrics
        from utils.plot import hist_value
        hist_value({'hits@1': metric_history['hits@1'],
                    'hits@3': metric_history['hits@3'],
                    'hits@10': metric_history['hits@10'],
                    'hits@100': metric_history['hits@100']},
                   path=path,
                   metric_name='hits@k',
                   name=model_id + 'valid_hits@k')
        hist_value({'mr': metric_history['mr']},
                   path=path,
                   metric_name='mr',
                   name=model_id + 'valid_mr')
        hist_value({'mrr': metric_history['mrr']},
                   path=path,
                   metric_name='mrr',
                   name=model_id + 'valid_mrr')
        hist_value({'loss': loss_history},
                   path=path,
                   metric_name='loss',
                   name=model_id + 'train_loss')
    # save train history
    data_to_save = metric_history
    data_to_save['loss'] = loss_history
    data_to_save['train_time'] = train_time
    data_to_save['evaluate_time'] = evaluate_time
    save_json(data_to_save, name='train_history', path=path)
    print('model (checkpoint) id : ' + model_id)


def evaluate(model, batch_size, model_id, data='test', filter_out=False):
    """
    evaluate model in test set or valid set
    :param model_id:
    :param filter_out:
    :param model: model
    :param batch_size: batch size
    :param data: dataset
    :return: None
    """
    metrics = model.test(batch_size=batch_size, dataset=data, filter_out=filter_out)
    for key in metrics.keys():
        print(key, ': ', metrics[key], ' |', end='')
    path = './checkpoint/' + model.name + '/' + model_id + '/'
    save_json(metrics, name='test_result', path=path)


def main(args):
    # set random seed
    set_seed(args.seed)
    # set floating point precision
    set_default_fp(args.fp)
    model_handle = ModelHandle(args.model)
    # set device
    device = set_device(args.gpu)
    # load checkpoint
    if args.checkpoint is not None:
        model = load_checkpoint(args.checkpoint, model_handle, args, device)
    else:
        # load data
        data = DataLoader(args.dataset, root_path='./data/', type=model_handle.get_type(args.model))
        data.load()
        data.to(device)
        # base model
        if args.config:
            base_model = model_handle.get_base_model(data)
        else:
            base_model = model_handle.get_default_base_model(data)
        # Optimizer
        opt = get_optimizer(args, base_model)
        # model
        model = model_handle.Model(base_model, data, opt)
        model.to(device)

    if args.test:
        # evaluate
        model_id = args.checkpoint
        if args.checkpoint is None:
            raise Exception("You need to load a checkpoint for testing!")
        evaluate(model, args.batch_size, model_id=model_id, filter_out=args.filter)
    else:
        # train
        train(model, args.epoch, args.batch_size, args.eva_step, args.early_stop, args.monitor, filter_out=args.filter,
              plot=args.plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGTL')
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="choose model")
    parser.add_argument("--config", action='store_true', default=False,
                        help="configure parameter")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path and name of model saved')
    # dataset
    parser.add_argument("--dataset", type=str, default=None,
                        help="choose dataset")
    parser.add_argument("--filter", action='store_true', default=False,
                        help="filter triplets. when a query (s,r,?) has multiple objects, filter out others.")
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
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # train
    parser.add_argument("--epoch", type=int, default=30,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size.")
    parser.add_argument("--eva-step", type=int, default=1,
                        help="evaluate model on valid set after 'eva-step' step of training.")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="patience for early stop.")
    parser.add_argument("--monitor", type=str, default='mrr',
                        help="monitor metric for early stop ")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="plot loss and metrics.")
    # test
    parser.add_argument("--test", action='store_true', default=False,
                        help="evaluate model on test set, and notice that you must load a checkpoint for this.")
    # other
    parser.add_argument("--fp", type=str, default='fp32',
                        help="Floating point precision (fp16, bf16, fp32 or fp64) ")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Use the GPU with the lowest memory footprint by default. "
                             "Specify a GPU by setting this parameter to a GPU id which equal to or greater than 0."
                             "Set this parameter to -1 to use the CPU."
                        )
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed.")
    args_parsed = parser.parse_args()

    main(args_parsed)
