# train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Setup project path (assumes you're running inside project root)
project_path = os.getcwd()
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")

# Load the census.csv data
data = pd.read_csv(data_path)

# Split the provided data into train and test datasets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
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

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save model and encoder artifacts
model_path = os.path.join(project_path, "model", "model.joblib")
save_model(model, model_path)

encoder_path = os.path.join(project_path, "model", "encoder.joblib")
save_model(encoder, encoder_path)

lb_path = os.path.join(project_path, "model", "lb.joblib")
save_model(lb, lb_path)

# Load the model (sanity check)
model = load_model(model_path)

# Run inference on test data
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall Test Set Performance -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute slice performance and write to slice_output.txt
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                test, col, slicevalue, cat_features, "salary", encoder, lb, model
            )
            if p is not None:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n", file=f)

print("✅ Model training complete. All artifacts and slice outputs saved.")