# model_scoring.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-08

import click
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

@click.command()
@click.option('--X-test', type=str, help="X_test dataframe")
@click.option('--y-test', type=str, help="y_test dataframe")
@click.option('--model', type=str, help="A fit model object")
def model_scoring(X_test, y_test, model):
    """
    Evaluates the given fitted model on the test data and returns the score results as a dataframe.

    Creates a DataFrame with 4 columns, one each for accuracy, F1 score, precision, recall. 

    Parameters:
    ----------
    x_test : pandas.DataFrame
        The DataFrame containing the X_test data
    y_test : pandas.DataFrame
        The DataFrame containing the y_test data
    model : a fitted model object
        The fit model to score on the test data

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with four columns:
        - 'accuracy': Lists the calculated accuracy of the given model on the test data.
        - 'f1': Lists the calculated F1 score of the given model on the test data.
        - 'precision': Lists the calculated precision of the given model on the test data
        - 'recall': Lists the calculated recall of the given model on the test data

    Raises:
    --------
    ValueError
        If the X_test and y_test paths do not end with '.csv'
    FileNotFoundError
        If the X_test, y_test, or pipeline_from paths do not exist.

    Examples:
    --------
    >>> import pandas as pd
    >>> import pickle
    >>> import DummyRegressor
    >>> X_train = pd.read_csv('x_train.csv') # Replace 'x_train.csv' with your X_train dataset file
    >>> y_train = pd.read_csv('y_train.csv') # Replace 'y_train.csv' with your y_train dataset file
    >>> X_test = pd.read_csv('x_test.csv') # Replace 'x_test.csv' with your X_test dataset file
    >>> y_test = pd.read_csv('y_test.csv') # Replace 'y_test.csv' with your y_test dataset file
    >>> model = DummyRegressor())
    >>> model.fit(X_train, y_train)
    >>> model_scoring('X_test', 'y_test', model)

    Notes:
    -----
    This function uses the scikit learn library to score the given model on test data. 

    """
    
    # Check input datatypes
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")
    if not isinstance(y_test, pd.Series):
        raise TypeError("X_test must be a pandas Series")

    # Check that the given X_test and y_test are not empty dataframes
    if X_test.empty:
        raise ValueError("X_test must contain observations.")
    if y_test.empty:
        raise ValueError("y_test must contain observations")
    
    # Compute and save the model scores
    accuracy = model.score(X_test, y_test)
    test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    precision = precision_score(y_test, model.predict(X_test), average='weighted')
    recall = recall_score(y_test, model.predict(X_test), average='weighted')

    results_table = pd.DataFrame({'accuracy': [accuracy],
                                  'f1': [test_f1],
                                  'precision': [precision],
                                  'recall': [recall]})
    return results_table

if __name__ == '__main__':
    model_scoring()