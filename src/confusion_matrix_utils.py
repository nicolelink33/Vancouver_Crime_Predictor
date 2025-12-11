# confusion_matrix_utils.py
# author: Tirth Joshi, Nicole Link, Zain Nofal
# date 2025-12-10

"""
Confusion matrix utilities for Vancouver Crime Predictor.

This module provides functions to create and save confusion matrix visualizations
for model evaluation.

Author: Tirth Joshi, Nicole Link, Zain Nofal
Date: 2025-12-10
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def create_confusion_matrix(y_test, y_pred, labels, title, save_path):
    """
    Create and save a confusion matrix plot.
    
    Parameters
    ----------
    y_test : pandas.Series or array-like
        True labels from the test set
    y_pred : array-like
        Predicted labels from the model
    labels : list
        Class labels for display on the confusion matrix
    title : str
        Title for the confusion matrix plot
    save_path : str
        File path where the confusion matrix plot will be saved
        
    Returns
    -------
    None
        Saves the confusion matrix plot to the specified path
        
    Raises
    ------
    ValueError
        If y_test or y_pred are empty
        If labels list is empty
        If save_path is not a valid string
    TypeError
        If y_test or y_pred are not array-like
        If labels is not a list
        If title or save_path are not strings
        
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.dummy import DummyClassifier
    >>> X_train = pd.DataFrame({'feature': [1, 2, 3, 4]})
    >>> y_train = pd.Series(['A', 'B', 'A', 'B'])
    >>> X_test = pd.DataFrame({'feature': [1, 2]})
    >>> y_test = pd.Series(['A', 'B'])
    >>> model = DummyClassifier()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> create_confusion_matrix(y_test, y_pred, ['A', 'B'], 
    ...                          'Test Confusion Matrix', 'cm.png')
    
    Notes
    -----
    This function uses matplotlib and scikit-learn's ConfusionMatrixDisplay
    to create a standardized confusion matrix visualization with a blue colormap.
    The plot is saved with high resolution (300 dpi) and tight bounding box.
    """
    
    # Input validation
    if not hasattr(y_test, '__len__'):
        raise TypeError("y_test must be array-like")
    if not hasattr(y_pred, '__len__'):
        raise TypeError("y_pred must be array-like")
    if not isinstance(labels, list):
        raise TypeError("labels must be a list")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(save_path, str):
        raise TypeError("save_path must be a string")
    
    if len(y_test) == 0:
        raise ValueError("y_test cannot be empty")
    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")
    if len(labels) == 0:
        raise ValueError("labels cannot be empty")
    if save_path.strip() == "":
        raise ValueError("save_path cannot be empty")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Customize plot
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Crime Type', fontsize=12)
    plt.ylabel('Actual Crime Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
