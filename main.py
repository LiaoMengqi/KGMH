from .utils import data_process
from .DataLoader import DataLoader

data = DataLoader("dataset", "../data")
data_process.split_data(data)
