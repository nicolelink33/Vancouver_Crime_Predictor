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
@click.option('--x-test-path', type=str, help="Path to X_test")
@click.option('--y-test-path', type=str, help="Path to y_test")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit model pipeline object lives")
@click.option('--results-to', type=str, help="Path to save table to")
def model_scoring(x_test_path, y_test_path, pipeline_from, results_to):
    """
    Evaluates the given fitted model on the test data and saves the score results.

    Creates a DataFrame with 4 columns, one each for accuracy, F1 score, precision, recall. 

    Parameters:
    ----------
    x_test_path : str
        The pathway to the csv file containing the X_test data (the features).
    y_test_path : str
        The pathway to the csv file containing the y_test data (the target).
    pipeline_from : str
        The pathway to the pickle file containing the fit model.
    results_to : str
        The pathway to save the resulting dataframe to. 

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with four columns:
        - 'accuracy': Lists the calculated accuracy of the given model on the test data.
        - 'f1': Lists the calculated F1 score of the given model on the test data.
        - 'precision': Lists the calculated precision of the given model on the test data
        - 'recall': Lists the calculated recall of the given model on the test data

    Examples:
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> import pickle
    >>> from sklearn.svm import LinearSVC
    >>> X_train = pd.read_csv('x_train.csv') # Replace 'x_train.csv' with your X_training dataset file
    >>> y_train = pd.read_csv('y_train.csv') # Replace 'y_train.csv' with your y_training dataset file
    >>> with open(preprocessor, 'rb') as f:  
    >>>     preprocessor_obj = pickle.load(f)  # Replace 'preprocessor' with your column transformer object
    >>> model_pipe = make_pipeline(preprocessor_obj, LinearSVC(C=1))
    >>> model_pipe.fix(X_train, y_train)
    >>> with open("results/models/model.pickle"), 'wb') as f:
    >>>     pickle.dump(model_pipe, f)
    >>> model_scoring('data/processed/X_test.csv', 'data/processed/y_test.csv', "results/models/model.pickle", "results/tables/model_score.csv")

    Notes:
    -----
    This function uses the scikit learn library to score the given model on test data. 

    """
    
    # Read in the data and fitted model
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    with open(pipeline_from, 'rb') as f:
        model = pickle.load(f)
    
    # Compute and save the model scores
    accuracy = model.score(X_test, y_test)
    test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    precision = precision_score(y_test, model.predict(X_test), average='weighted')
    recall = recall_score(y_test, model.predict(X_test), average='weighted')

    results_table = pd.DataFrame({'accuracy': [accuracy],
                                  'f1': [test_f1],
                                  'precision': [precision],
                                  'recall': [recall]})
    results_table.to_csv(results_to, index=False)

if __name__ == '__main__':
    model_scoring()