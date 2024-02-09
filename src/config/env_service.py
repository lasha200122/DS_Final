import os
from dotenv import load_dotenv


load_dotenv()

def get_train_data_path():
    return os.getenv('TRAIN_DATA_PATH', 'data/raw/train.csv')

def get_test_data_path():
    return os.getenv('TEST_DATA_PATH', 'data/raw/test.csv')

def get_processed_data_path():
    return os.getenv('PROCESSED_DATA_PATH', 'data/processed/')

def get_model_save_path():
    return os.getenv('MODEL_SAVE_PATH', 'outputs/models/')

def get_predictions_save_path():
    return os.getenv('PREDICTIONS_SAVE_PATH', 'outputs/predictions/')
