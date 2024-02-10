import joblib
from config.env_service import get_train_data_path, get_processed_data_path, get_vectorizer_type, get_processed_data_file, get_test_data_path, get_processed_test_file
from utils.text_preprocessing import preprocess_text, vectorize_texts
import pandas as pd
import os
from scipy.sparse import save_npz, hstack
import numpy as np


class DataLoader:
    def __init__(self, data_path=None, processed_data_path=None):
        self.data_path = data_path if data_path is not None else get_train_data_path()
        self.processed_data_path = processed_data_path if processed_data_path is not None else get_processed_data_path()
        self.vectorizer = None
        self.vectorizer_type = get_vectorizer_type()

    def load_data(self):
        train_data = pd.read_csv(self.data_path)
        return train_data
    
    def save_vectorizer(self):
        vectorizer_path = os.path.join(self.processed_data_path, 'vectorizer.joblib')
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    def preprocess_and_vectorize_data(self, data, is_train_data=True):
        data['processed_text'] = data['review'].apply(preprocess_text)
        texts = data['processed_text'].tolist()

        if is_train_data:
            vectorized_data, self.vectorizer = vectorize_texts(texts, vectorizer_type=self.vectorizer_type)
            
            self.save_vectorizer()
        else:
            vectorized_data = self.vectorizer.transform(texts)

        if 'sentiment' in data:
            data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0}).astype(int)
            if isinstance(vectorized_data, np.ndarray):
                vectorized_data = np.hstack((vectorized_data, data['sentiment'].values.reshape(-1, 1)))
            else:
                vectorized_data = hstack((vectorized_data, data['sentiment'].values.reshape(-1, 1)))

        return vectorized_data

    def save_processed_data(self, vectorized_data, file):
        processed_data_file = os.path.join(self.processed_data_path, file)
        save_npz(processed_data_file, vectorized_data)
        print(f"Vectorized data saved to {processed_data_file}")

    def get_vectorizer(self):

        return self.vectorizer


if __name__ == "__main__":
    train_loader = DataLoader()
    train_raw_data = train_loader.load_data()
    vectorized_train_data = train_loader.preprocess_and_vectorize_data(train_raw_data, is_train_data=True)
    train_loader.save_processed_data(vectorized_train_data, get_processed_data_file())

    vectorizer_path = os.path.join(train_loader.processed_data_path, 'vectorizer.joblib')
    fitted_vectorizer = joblib.load(vectorizer_path)

    test_loader = DataLoader(get_test_data_path(), train_loader.processed_data_path)
    test_loader.vectorizer = fitted_vectorizer
    test_raw_data = test_loader.load_data()
    vectorized_test_data = test_loader.preprocess_and_vectorize_data(test_raw_data, is_train_data=False)
    test_loader.save_processed_data(vectorized_test_data, get_processed_test_file())
    