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
    '''Evaluates the inputted fit model on the test data 
    and saves the score results.'''
    
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