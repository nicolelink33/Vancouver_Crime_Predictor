# test_scoring.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-08

import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from src.model_scoring import model_scoring

# Create data, files, models for tests:

# Create simple test data with accuracy = 1
X_1 = pd.DataFrame({"values": [1, 1, 1, 1]})
y_1 = pd.Series([1, 1, 1, 1], name="values")
output_acc1 = pd.DataFrame({'accuracy': [1.0],
                        'f1': [1.0],
                        'precision': [1.0],
                        'recall': [1.0]})

# Edge case: Create test data with accuracy = 0
X_0 = pd.DataFrame({"values": [0, 0, 0, 0]})
y_0 = pd.Series([0, 0, 0, 0], name="values")
output_acc0 = pd.DataFrame({'accuracy': [0.0],
                        'f1': [0.0],
                        'precision': [0.0],
                        'recall': [0.0]})


# Edge case: Create test data with no observations
X_empty = pd.DataFrame({'values': []})
y_empty = pd.Series([], name="values")


# Create invalid datatypes for X_test and y_test
X_tuple = (1, 1, 1, 1)
y_tuple = (1, 1, 1, 1)

# Create an un-fit dummy model:
unfit_dummy = DummyClassifier(strategy='most_frequent')

# Train dummy model to predict class 1 and save model
X_train = pd.DataFrame({"values": [1, 1, 1, 1]})
y_train = pd.Series([1, 1, 1, 1], name="values")
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)


# Tests:

# Error case: `model_scoring` should throw an error if the X_test and y_test don't exist.
# Error case: `model_scoring` should throw an error when incorrect types are passed to X_test and y_test
def test_scoring_df_errors():
    with pytest.raises(NameError):
        model_scoring(fake_X_test,  # nonexistent object
                      fake_y_test,  # nonexistent object
                      dummy)
    with pytest.raises(TypeError):
        model_scoring(X_tuple,    # incorrect data format
                      y_tuple,    # incorrect data format
                      dummy)

# Error case: `model_scoring` should throw an error if the model does not exist or is not fitted
def test_scoring_unfit_error():
    with pytest.raises(NameError):
        model_scoring(X_1,
                      y_1,
                      fake_dummy)  # nonexistent object
    with pytest.raises(NotFittedError):
        model_scoring(X_1,
                      y_1,
                      unfit_dummy)  # unfit model

# Error case: `model_scoring` should throw an error if given an empty X_test and y_test
def test_scoring_empty():
    with pytest.raises(ValueError):
        model_scoring(X_empty,
                     y_empty,
                     dummy)

# Unit test: Test X_1 and y_1, which should give accuracy=1
def test_scoring_1():
    # Run model_scoring
    model_output = model_scoring(X_1,
                  y_1,
                 dummy)

    # Check that model output is equal to output_acc1
    pd.testing.assert_frame_equal(model_output, output_acc1)

# Test X_0 and y_0, which should give accuracy = 0 
def test_scoring_0():
    # Run model_scoring
    model_output = model_scoring(X_0,
                                 y_0,
                                 dummy)

    # Check that model output is equal to output_acc0
    pd.testing.assert_frame_equal(model_output, output_acc0)


