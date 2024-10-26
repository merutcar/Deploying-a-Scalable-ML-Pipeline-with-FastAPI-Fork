import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

"""
The original data will be imported and a subset will be used.
The subset will be processed and then used for testing.
"""

# Load a small subset of the actual data for testing
data_path = "data/census.csv"
data = pd.read_csv(data_path)
sample_data = data.head(10)

# Split the data
train, test = train_test_split(sample_data, test_size=0.20, random_state=42)

# Declaring features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Process data for testing
X_train, y_train, encoder, lb = process_data(train,
    categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(test,
    categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# TEST 1
def test_model_type():
    """
    Tests if the trained model is a RandomForestClassifier.

    INPUT: None
    OUTPUT: None

    First, model is trained with training data.
    Then, the assertion is tested.
    """

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TEST 2
def test_model_metrics_dtype():
    """
    Tests if the model metrics computed are returned as floats.

    INPUT: None
    OUTPUT: None

    First, it trains the model.
    Second, it predicts based on the model.
    Then, it calculates the metrics as it normally would.
    Lastly, three assertions are called, one for each metric.
    """

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


# TEST 3
# Test if the sizes of training and test datasets are as expected
def test_train_test_split_isempty():
    """
    Tests if the training and test datasets are divided correctly so
    that each contains some data.

    INPUT: None
    OUTPUT: None

    Two assertions are called. One for the training data set and another
    for the testing dataset. The length is tried.
    """

    assert len(train) > 0
    assert len(test) > 0
