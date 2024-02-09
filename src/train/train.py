from src.data_loader import DataLoader
from src.utils.text_preprocessing import preprocess_text, vectorize_texts


class Trainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None  # Initialize your model here, e.g., a scikit-learn classifier

    def preprocess_and_vectorize(self):
        data = self.data_loader.load_data()
        data['processed_review'] = data['review'].apply(preprocess_text)
        X, self.vectorizer = vectorize_texts(data['processed_review'].tolist(), vectorizer_type='tfidf')
        y = data['sentiment']
        return X, y

    def train(self, X, y):
        self.model.fit(X, y)


if __name__ == "__main__":
    data_loader = DataLoader()
    trainer = Trainer(data_loader)
    X_train, y_train = trainer.preprocess_and_vectorize()
    trainer.train(X_train, y_train)
