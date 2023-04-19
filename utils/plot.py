import matplotlib.pyplot as plt
import os


def hist_value(hist_dict: dict,
               path='./result_img/',
               name='metric'):
    if not os.path.exists(path):
        # create new directory
        os.mkdir(path)
    fig = plt.figure(num=1, figsize=(6, 4))
    ax = fig.add_subplot(111)
    for key in hist_dict.keys():
        ax.plot(hist_dict[key], label=key)
    ax.set_ylabel(name)
    ax.legend()
    ax.legend(loc='best')
    plt.savefig(path + name, dpi=600)
    plt.show()
