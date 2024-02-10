import numpy as np
from scipy.sparse import load_npz, coo_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from dotenv import load_dotenv
import xgboost as xgb


load_dotenv()


PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH")
PROCESSED_DATA_FILE = os.getenv("PROCESSED_DATA_FILE")
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")
RANDOM_FOREST= os.getenv("RANDOM_FOREST")
LOGISTIC_REGRESSION = os.getenv("LOGISTIC_REGRESSION")
XGB_MODEL = os.getenv("XGB_MODEL")


def load_vectorized_data():
    file_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_FILE)
    vectorized_data = load_npz(file_path)
    if isinstance(vectorized_data, coo_matrix):
        vectorized_data = vectorized_data.tocsr()
    return vectorized_data

def train_model(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(TEST_SIZE), random_state=int(RANDOM_STATE))

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return clf

def save_model(model, filename):
    file_path = os.path.join(MODEL_SAVE_PATH, filename)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


if __name__ == "__main__":
    vectorized_data = load_vectorized_data()
    X = vectorized_data[:, :-1]
    y = vectorized_data[:, -1].toarray().ravel()

    y = y.astype(int)

    unique_classes = np.unique(y)
    print("Unique classes in y:", unique_classes)

    rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    trained_rf = train_model(rf_clf, X, y)
    save_model(trained_rf, RANDOM_FOREST)

    lr_clf = LogisticRegression(random_state=RANDOM_STATE)
    trained_lr = train_model(lr_clf, X, y)
    save_model(trained_lr, LOGISTIC_REGRESSION)

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    trained_xgb = train_model(xgb_clf, X, y)
    save_model(trained_xgb, XGB_MODEL)
