from config.env_service import get_train_data_path
import pandas as pd

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path if data_path is not None else get_train_data_path()

    def load_data(self):
        train_data = pd.read_csv(self.data_path)
        return train_data
