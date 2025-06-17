# filename: test_ml.py

import pytest
import pandas as pd
import os
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Load sample data
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "age": [25, 32],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "HS-grad"],
        "education-num": [7, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Farming-fishing"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 60],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })
    return data

def test_process_data(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == 2
    assert len(y) == 2

def test_train_model(sample_data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)

def test_compute_model_metrics():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1