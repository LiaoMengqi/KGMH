from data_loader import DataLoader
from utils.data_process import split_data_by_time
from models.layers import GNNLayer

data = DataLoader("ICEWS14s", "./data")
data.load(load_time=True)
data_train_split, _, _ = split_data_by_time(data.train)

