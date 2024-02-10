import numpy as np
import joblib
from scipy.sparse import load_npz
import os
from dotenv import load_dotenv


load_dotenv()

MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")
TEST_DATA_FILE = os.getenv("PROCESSED_TEST_FILE")
RANDOM_FOREST = os.getenv("RANDOM_FOREST")
LOGISTIC_REGRESSION = os.getenv("LOGISTIC_REGRESSION")
XGB_MODEL = os.getenv("XGB_MODEL")
PREDICTIONS_SAVE_PATH = os.getenv("PREDICTIONS_SAVE_PATH")

def load_vectorizer(filename):
    """
    Load the vectorizer from the specified file.
    
    :param filename: The name of the file where the vectorizer is saved.
    :return: The loaded vectorizer.
    """
    vectorizer_path = os.path.join(PROCESSED_DATA_PATH, filename)
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

def load_model(filename):
    file_path = os.path.join(MODEL_SAVE_PATH, filename)
    model = joblib.load(file_path)
    return model

def load_test_data():
    file_path = os.path.join(PROCESSED_DATA_PATH, TEST_DATA_FILE)
    test_data = load_npz(file_path).tocsr() 
    return test_data

def predict(model, X):
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    vectorizer = load_vectorizer('vectorizer.joblib')  # Ensure you have this function or just use joblib.load with the correct path

    # Load models
    rf_model = load_model(RANDOM_FOREST)
    lr_model = load_model(LOGISTIC_REGRESSION)
    xgb_model = load_model(XGB_MODEL)

    # Load and transform test data
    X_test = load_test_data()  # Make sure this loads the raw test data if you need to preprocess it

    # Make predictions
    rf_predictions = predict(rf_model, X_test)
    lr_predictions = predict(lr_model, X_test)
    xgb_predictions = predict(xgb_model, X_test)
    np.savetxt(os.path.join(PREDICTIONS_SAVE_PATH, 'rf_predictions.csv'), rf_predictions, delimiter=',')
    np.savetxt(os.path.join(PREDICTIONS_SAVE_PATH, 'lr_predictions.csv'), lr_predictions, delimiter=',')
    np.savetxt(os.path.join(PREDICTIONS_SAVE_PATH, 'xgb_predictions.csv'), xgb_predictions, delimiter=',')
