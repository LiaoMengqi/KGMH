from utils.data_process import load_data
from utils.data_process import load_dict
from utils.data_process import reverse_dict


class DataLoader(object):
    def __init__(self, dataset, path):
        """
        param dataset: the name of dataset
        param path: the path of dataset
        """
        self.path = path
        self.dataset = dataset

        # data
        self.train = None
        self.valid = None
        self.test = None
        self.id2entity = {}
        self.id2relation = {}
        self.entity2id = {}
        self.relation2id = {}
        self.num_relation = 0
        self.num_entity = 0

    def load(self, load_time=False, encoding='utf-8'):
        """
        Initialize training set, validation set and test set.
        """
        file = self.path + '/' + self.dataset + '/train.txt'
        self.train = load_data(file, load_time=load_time, encoding=encoding)

        file = self.path + '/' + self.dataset + '/valid.txt'
        self.valid = load_data(file, load_time=load_time, encoding=encoding)

        file = self.path + '/' + self.dataset + '/test.txt'
        self.test = load_data(file, load_time=load_time, encoding=encoding)

        file = self.path + '/' + self.dataset + '/relation2id.txt'
        self.relation2id = load_dict(file)
        self.id2relation = reverse_dict(self.relation2id)
        self.num_relation = len(self.relation2id)

        file = self.path + '/' + self.dataset + '/entity2id.txt'
        self.entity2id = load_dict(file)
        self.id2entity = reverse_dict(self.id2entity)
        self.num_entity = len(self.entity2id)

    def to(self, device):
        if device == 'cpu':
            self.train.cpu()
            self.valid.cpu()
            self.test.cpu()
        elif device == 'cuda':
            self.train = self.train.cuda()
            self.valid = self.valid.cuda()
            self.test = self.test.cuda()
