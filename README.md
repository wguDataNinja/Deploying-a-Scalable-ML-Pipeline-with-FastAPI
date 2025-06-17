# Deploying a Scalable ML Pipeline with FastAPI

**Project GitHub Repo:** https://github.com/wguDataNinja/Deploying-a-Scalable-ML-Pipeline-with-FastAPI
## Project Overview

This project builds and deploys a machine learning pipeline to predict income levels using U.S. Census data. It covers the full ML lifecycle: data processing, model training, model evaluation (including slice metrics), API deployment, automated testing, and CI/CD integration.


## Setup

1. Clone the repository:

   git clone https://github.com/wguDataNinja/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
   cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI

2. Create a virtual environment and install requirements:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

## Model Training

To train the model and generate artifacts:

   python train_model.py

- Trains a RandomForestClassifier on data/census.csv.
- Saves model artifacts (model.joblib, encoder.joblib, lb.joblib) to the model/ directory.
- Generates slice evaluation metrics to slice_output.txt.

## Running the API

Start the FastAPI server:

   uvicorn main:app --reload

> Note: Keep this terminal running. Open a new terminal window to test the API while the server stays running.

### Testing the API locally

Run the local API test client:

   python local_api.py

- Sends GET and POST requests to verify API functionality.
- Sample POST request uses a sample census record for inference.

## Testing

To run all unit tests:

   pytest

Tests include:

- Data preprocessing (test_ml.py)
- Model inference and metrics (test_ml.py)
- API endpoints (test_api.py)

## CI/CD

GitHub Actions is configured to automatically run:

- pytest (unit tests)
- flake8 (code linting)

CI/CD status verified as part of capstone submission (continuous_integration.png).

## Author

- Buddy Owens
- WGU D501 ML Ops Project 2
- June 2025