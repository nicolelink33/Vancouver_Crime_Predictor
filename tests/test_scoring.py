# test_scoring.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-08

import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

# Create data, files, models for tests:

# Create simple test data with accuracy = 1 and save to csv files
X_1 = pd.DataFrame({"values": [1, 1, 1, 1]})
y_1 = pd.DataFrame({"values": [1, 1, 1, 1]})
output_acc1 = pd.DataFrame({'accuracy': [1],
                        'f1': [1],
                        'precision': [1],
                        'recall': [1]})
X_1.to_csv("tests/files/X_1.csv")
y_1.to_csv("tests/files/y_1.csv")

# Edge case: Create test data with accuracy = 0
X_0 = pd.DataFrame({"values": [0, 0, 0, 0]})
y_0 = pd.DataFrame({"values": [0, 0, 0, 0]})
output_acc0 = pd.DataFrame({'accuracy': [0],
                        'f1': [0],
                        'precision': [0],
                        'recall': [0]})
X_0.to_csv("tests/files/X_0.csv")
y_0.to_csv("tests/files/y_0.csv")

# Edge case: Create test data with no observations
X_empty = pd.Dataframe({'values': []})
y_empty = pd.Dataframe({'values': []})
X_empty.to_csv("tests/files/X_empty.csv")
y_empty.to_csv("tests/files/y_empty.csv")

# Create invalid txt files for X_test and y_test
X_1.to_csv("tests/files/X_1.txt", sep='\t')
y_1.to_csv("tests/files/y_1.txt", sep='\t')

# Create and save an un-fit dummy model:
unfit_dummy = DummyClassifier(strategy='most_frequent')
with open("tests/files/unfit_dummy.pickle", 'wb') as f:
    pickle.dump(unfit_dummy, f)

# Train dummy model to predict class 1 and save model
X_train = pd.DataFrame({"values": [1, 1, 1, 1]})
y_train = pd.DataFrame({"values": [1, 1, 1, 1]})
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
with open("tests/files/dummy.pickle", 'wb') as f:
    pickle.dump(dummy, f)


# Tests:

# Error case: `model_scoring` should throw an error if the X_test and y_test don't exist or invalid paths are given.
# Error case: `model_scoring` should throw an error when incorrect types are passed to X_test and y_test
def test_scoring_file_errors():
    with pytest.raises(FileNotFoundError):
        model_scoring('data/processed/fake_X.csv',  # nonexistent file
                      'data/processed/fake_y.csv',  # nonexistent file
                      'tests/files/dummy.pickle',
                      'tests/files/test_output.csv')
    with pytest.raises(ValueError):
        model_scoring("tests/files/X_1.txt",    # incorrect file format (tab-separated)
                      "tests/files/y_1.txt",    # incorrect file format (tab-separated)
                      "tests/files/dummy.pickle",
                      'tests/files/test_output.csv')

# Error case: `model_scoring` should throw an error if the model does not exist or is not fitted
def test_scoring_unfit_error():
    with pytest.raises(FileNotFoundError):
        model_scoring("tests/files/X_1.csv",
                      "tests/files/y_1.csv",
                      "tests/files/fake_model.pickle",  # nonexistent file
                      'tests/files/test_output.csv')
    with pytest.raises(NotFittedError):
        model_scoring("tests/files/X_1.csv",
                      "tests/files/y_1.csv",
                      "tests/files/unfit_dummy.pickle",  # unfit model
                      'tests/files/test_output.csv')

# Error case: `model_scoring` should throw an error if given an empty X_test and y_test
def test_scoring_empty_csv():
    with pytest.raises(ValueError):
        model_scoring("tests/files/X_empty.csv",
                      "tests/files/y_empty.csv",
                      "tests/files/dummy.pickle",
                      "tests/files/test_output.csv")

# Unit test: Test X_1 and y_1, which should give accuracy=1
def test_scoring_1():
    # Run model_scoring
    model_scoring("tests/files/X_1.csv",
                  "tests/files/y_1.csv",
                  "tests/files/dummy.pickle",
                  "tests/files/output_acc1.csv")
    
    # Read in result of model_scoring
    model_output = pd.read_csv("tests/files/output_acc1.csv")

    # Check that model output is equal to output_acc1
    assert pd.testing.assert_frame_equal(model_output, output_acc1)

# Test X_0 and y_0, which should give accuracy = 0 
def test_scoring_0():
    # Run model_scoring
    model_scoring("tests/files/X_0.csv",
                  "tests/files/y_0.csv",
                  "tests/files/dummy.pickle",
                  "tests/files/output_acc0.csv")
    
    # Read in result of model_scoring
    model_output = pd.read_csv("tests/files/output_acc0.csv")

    # Check that model output is equal to output_acc1
    assert pd.testing.assert_frame_equal(model_output, output_acc0)

