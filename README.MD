### This project has a modular structure, where each folder has a specific duty.

```
MLE Architecture
├── data
|   |── processed
|   |   |── processed_test.npz
|   |   |── processed_train.npz
|   |   |__ vectorizer.joblib
|   |── raw
|   |   |──test.csv
|   |   |__train.csv
|── notebooks
|── outputs
|   |── figures
|   |── models
|   |   |── logistic_regression.pkl
|   |   |── random_forest.pkl
|   |   |__ xgb.pkl
|   |__ predictions
|── src
|   |── config
|   |   |__ env_service.py
|   |── inference
|   |   |── __init__.py
|   |   |── Dockerfile
|   |   |__ run_inference.py
|   |── train
|   |   |── __init__.py
|   |   |── Dockerfile
|   |   |__ train.py
|   |── utils
|   |   |__ text_preprocessing.py
|   |__ data_loader.py
|── .env
|── .gitignore
|── .README.MD
|__ requirements.txt
```

# Project Structure

The project is organized into distinct folders, each designed for specific roles within the MLE lifecycle:

- data/: Contains datasets used throughout the project.
  - processed/: Stores preprocessed data and the vectorizer object.
  - raw/: Houses the original, unprocessed datasets.
- notebooks/: Jupyter notebooks for exploratory data analysis and initial prototyping.
- outputs/: Directory for storing output artifacts.
  - figures/: Generated plots and figures.
  - models/: Trained model files.
  - predictions/: Results from model predictions.
- src/: Core source code for the project.
  - config/: Configuration settings and utilities.
  - inference/: Inference scripts and Dockerfile for deployment.
  - train/: Training scripts and Dockerfile for model training environments.
  - utils/: Utility functions, including text preprocessing scripts.
- .env: Environment variables defining path settings and configurations.
- requirements.txt: Project dependencies for replication.

# Setup Instructions

1. Environment Setup: Ensure Python 3.6+ is installed. Create a virtual environment and activate it:

```powershell
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
```

2. Install Dependencies: Install Python packages from requirements.txt:

```powershell
pip install -r requirements.txt
```

3. Environment Variables: Set up the .env file according to your environment specifics. This file should include paths to data, model, and prediction output directories, among other configurations.

# Data Preprocessing

To preprocess your data, run the data_loader.py script within the src/ directory. This script reads the raw datasets, applies preprocessing, vectorizes the text data, and saves the processed data for training and inference:

```powershell
python src/data_loader.py
```

This script also saves the text vectorizer to ensure consistency between training and inference phases.

# Model Training

Navigate to src/train/ and run the train.py script to start training your models:

```powershell
cd src/train
python train.py
```

Trained models will be saved to the directory specified by MODEL_SAVE_PATH in your .env file.

# Running Inference

For inference, use the run_inference.py script located in src/inference/:

```powershell
cd src/inference
python run_inference.py
```

Predictions will be output to the directory defined by PREDICTIONS_SAVE_PATH in your .env file.

# Docker Support

Dockerfiles for both training and inference are provided for containerization purposes. To build and run these containers, use the following commands:

- Training Container:

```Docker
docker build -f ./src/train/Dockerfile --build-arg settings_name=.env -t training_image .

docker run -it training_image /bin/bash
```

- Inference Container

```Docker
docker build -f ./src/inference/Dockerfile --build-arg settings_name=.env -t inference_image .

docker run -it inference_image /bin/bash
```
