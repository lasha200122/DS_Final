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

def get_vectorizer_type():
    return os.getenv('VECTORIZER_TYPE', 'tfidf')

def get_processed_data_file():
    return os.getenv('PROCESSED_DATA_FILE', 'processed_train.npz')

def get_processed_test_file():
    return os.getenv('PROCESSED_TEST_FILE', 'processed_test.npz')
