# test_confusion_matrix.py
# author: Tirth Joshi, Nicole Link, Zain Nofal
# date 2025-12-10

"""
Tests for confusion matrix utility functions.

This module contains unit tests and error tests for the create_confusion_matrix
function used in the Vancouver Crime Predictor project.

Author: Tirth Joshi, Nicole Link, Zain Nofal
Date: 2025-12-10
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.dummy import DummyClassifier
from src.confusion_matrix_utils import create_confusion_matrix


# Test data setup
X_train = pd.DataFrame({'feature': [1, 2, 3, 4, 5, 6]})
y_train = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
X_test = pd.DataFrame({'feature': [1, 2, 3, 4]})
y_test_perfect = pd.Series(['A', 'B', 'A', 'B'])
y_pred_perfect = pd.Series(['A', 'B', 'A', 'B'])
y_pred_wrong = pd.Series(['B', 'A', 'B', 'A'])

labels = ['A', 'B']


# Error Tests

def test_empty_y_test():
    """Test that empty y_test raises ValueError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(ValueError, match="y_test cannot be empty"):
        create_confusion_matrix(
            pd.Series([]), 
            y_pred_perfect, 
            labels, 
            'Test', 
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_empty_y_pred():
    """Test that empty y_pred raises ValueError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(ValueError, match="y_pred cannot be empty"):
        create_confusion_matrix(
            y_test_perfect, 
            pd.Series([]), 
            labels, 
            'Test', 
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_empty_labels():
    """Test that empty labels raises ValueError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(ValueError, match="labels cannot be empty"):
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_perfect, 
            [], 
            'Test', 
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_empty_save_path():
    """Test that empty save_path raises ValueError"""
    with pytest.raises(ValueError, match="save_path cannot be empty"):
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_perfect, 
            labels, 
            'Test', 
            ''
        )


def test_invalid_y_test_type():
    """Test that non-array-like y_test raises TypeError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(TypeError, match="y_test must be array-like"):
        create_confusion_matrix(
            123,  # Not array-like
            y_pred_perfect, 
            labels, 
            'Test', 
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_invalid_labels_type():
    """Test that non-list labels raises TypeError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(TypeError, match="labels must be a list"):
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_perfect, 
            ('A', 'B'),  # Tuple instead of list
            'Test', 
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_invalid_title_type():
    """Test that non-string title raises TypeError"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    with pytest.raises(TypeError, match="title must be a string"):
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_perfect, 
            labels, 
            123,  # Not a string
            tmp_path
        )
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# Unit Tests

def test_perfect_predictions():
    """Test confusion matrix with perfect predictions"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_perfect, 
            labels, 
            'Perfect Predictions', 
            tmp_path
        )
        
        # Check that file was created
        assert os.path.exists(tmp_path), "Confusion matrix file was not created"
        assert os.path.getsize(tmp_path) > 0, "Confusion matrix file is empty"
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_all_wrong_predictions():
    """Test confusion matrix with all wrong predictions"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        create_confusion_matrix(
            y_test_perfect, 
            y_pred_wrong, 
            labels, 
            'Wrong Predictions', 
            tmp_path
        )
        
        # Check that file was created
        assert os.path.exists(tmp_path), "Confusion matrix file was not created"
        assert os.path.getsize(tmp_path) > 0, "Confusion matrix file is empty"
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_with_real_model():
    """Test confusion matrix with a real trained model"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Train a dummy classifier
        model = DummyClassifier(strategy='most_frequent')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        create_confusion_matrix(
            y_test_perfect, 
            y_pred, 
            labels, 
            'Dummy Classifier', 
            tmp_path
        )
        
        # Check that file was created
        assert os.path.exists(tmp_path), "Confusion matrix file was not created"
        assert os.path.getsize(tmp_path) > 0, "Confusion matrix file is empty"
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_multiclass_predictions():
    """Test confusion matrix with multiple classes"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        y_test_multi = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'])
        y_pred_multi = pd.Series(['A', 'B', 'C', 'B', 'A', 'C'])
        labels_multi = ['A', 'B', 'C']
        
        create_confusion_matrix(
            y_test_multi, 
            y_pred_multi, 
            labels_multi, 
            'Multiclass Test', 
            tmp_path
        )
        
        # Check that file was created
        assert os.path.exists(tmp_path), "Confusion matrix file was not created"
        assert os.path.getsize(tmp_path) > 0, "Confusion matrix file is empty"
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
