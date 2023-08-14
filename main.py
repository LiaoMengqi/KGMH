import argparse
from model_handle import *
from utils.func import save_json
from utils.func import set_seed
from utils.func import set_default_fp
from utils.optm import get_optimizer
import time


def train(model, epochs, batch_size, step, early_stop, filter_out=False, plot=False):
    """
    train model
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

            for key in metrics.keys():
                print(key, ': ', metrics[key], ' |', end='')
                if key not in metric_history.keys():
                    metric_history[key] = []
                    metric_history[key].append(metrics[key])
                else:
                    metric_history[key].append(metrics[key])
            print('time: %f' % (time_end - time_start))
            if metric_history['mrr'][-1] < metric_history['mrr'][best]:
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
    print("best : hits@1: %f |hits@3: %f |hits@10: %f |hits@100: %f |mr: %f |mrr: %f" %
          (metric_history['hits@1'][best],
           metric_history['hits@3'][best],
           metric_history['hits@10'][best],
           metric_history['hits@100'][best],
           metric_history['mr'][best],
           metric_history['mrr'][best],))
    if plot:
        # plot loss and metrics
        from utils.plot import hist_value
        hist_value({'hits@1': metric_history['hits@1'],
                    'hits@3': metric_history['hits@3'],
                    'hits@10': metric_history['hits@10'],
                    'hits@100': metric_history['hits@100']},
                   value='hits@k',
                   name=model_id + '_valid_hits@k')
        hist_value({'mr': metric_history['mr']},
                   value='mr',
                   name=model_id + '_valid_mr')
        hist_value({'mrr': metric_history['mrr']},
                   value='mrr',
                   name=model_id + '_valid_mrr')
        hist_value({'loss': loss_history},
                   value='loss',
                   name=model_id + '_valid_loss')
    # save train history
    data_to_save = metric_history
    data_to_save['loss'] = loss_history
    data_to_save['train_time'] = train_time
    data_to_save['evaluate_time'] = evaluate_time
    path = './checkpoint/' + model.name + '/' + model_id + '/'
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
    # choose device
    if args.gpu != -1:
        if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            device = 'cpu'
        else:
            device = 'cuda:%d' % args.gpu
    else:
        device = 'cpu'
    # load checkpoint
    if args.checkpoint is not None:
        model = load_checkpoint(args.checkpoint, model_handle, args, device)
    else:
        # load data
        data = DataLoader(args.dataset, './data/temporal/extrapolation')
        data.load(load_time=True)
        data.to(device)
        # base model
        if args.config:
            base_model = model_handle.get_base_model(args, data)
        else:
            base_model = model_handle.get_default_base_model(args.model, data)
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
        train(model, args.epoch, args.batch_size, args.eva_step, args.early_stop, filter_out=args.filter,
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
    parser.add_argument("--epoch", type=int, default=15,
                        help="learning rate")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size.")
    parser.add_argument("--eva-step", type=int, default=1,
                        help="evaluate model on valid set after 'eva-step' step of training.")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="patience for early stop.")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="plot loss and metrics.")
    # test
    parser.add_argument("--test", action='store_true', default=False,
                        help="evaluate model on test set, and notice that you must load a checkpoint for this.")
    # other
    parser.add_argument("--fp", type=str, default='fp32',
                        help="floating point precision (fp16, fp32 or fp64) ")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="use GPU.")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed.")
    args_parsed = parser.parse_args()

    main(args_parsed)
