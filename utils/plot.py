import matplotlib.pyplot as plt
import os


def hist_value(hist_dict: dict,
               path='./',
               metric_name='value',
               name='model_name'):
    """
    plot and save
    :param metric_name:
    :param hist_dict:
    :param path: save path
    :param name: file name
    :return:
    """
    if not os.path.exists(path):
        # create new directory
        os.mkdir(path)
    fig = plt.figure(num=1, figsize=(6, 4))
    ax = fig.add_subplot(111)
    for key in hist_dict.keys():
        ax.plot(hist_dict[key], label=key)
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.legend(loc='best')
    plt.savefig(path + name, dpi=600)
    plt.show()
