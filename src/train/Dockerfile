# Dockerfile for training
# Start from a base Python image
FROM python:3.9-slim

# Initialize settings_name variable with the default value
ARG settings_name=.env
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

COPY outputs/ /app/outputs

# Copy the code itself
COPY src/train /app/src/train

# Copy utils.py
COPY src/utils /app/src/utils
COPY ${CONF_PATH} /app

RUN pip cache purge

RUN apt-get update && \
    apt-get install -y build-essential python3-dev

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 src/train/train.py