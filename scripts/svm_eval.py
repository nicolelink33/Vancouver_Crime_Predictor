# svm_eval.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-05

"""
Support Vector Machine model evaluation script for Vancouver Crime Predictor.

This module evaluates the trained SVM model on test data, generating performance
metrics, confusion matrices, and classification reports for both baseline and optimized models.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-05
"""

import click
import pickle
import os
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report
from src.model_scoring import model_scoring
from src.confusion_matrix_utils import create_confusion_matrix

@click.command()
@click.option('--x-test-path', type=str, help="Path to X_test")
@click.option('--y-test-path', type=str, help="Path to y_test")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the tables will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plots will be written to")
def svm_eval(x_test_path, y_test_path, pipeline_from, results_to, plot_to):
    '''Evaluates the Vancouver Crime Predictor on the test data 
    and saves the evaluation results.'''
    
    # Read in the data, baseline model, and fitted svm model
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    with open(os.path.join(pipeline_from, "svm_baseline_fit.pickle"), 'rb') as f:
        svm_base_fit = pickle.load(f)

    with open(os.path.join(pipeline_from, "svm_final_best_fit.pickle"), 'rb') as f:
        svm_fit = pickle.load(f)
    
    # Compute and save the baseline model scores
    baseline_results = model_scoring(X_test, y_test, svm_base_fit)
    baseline_results.to_csv(os.path.join(results_to, "tables", "svm_baseline_score.csv"), index=False)

    # base_accuracy = svm_base_fit.score(X_test, y_test)
    # base_f1 = f1_score(y_test, svm_base_fit.predict(X_test), average='weighted')
    # base_precision = precision_score(y_test, svm_base_fit.predict(X_test), average='weighted')
    # base_recall = recall_score(y_test, svm_base_fit.predict(X_test), average='weighted')

    # results_table = pd.DataFrame({'accuracy': [base_accuracy],
    #                               'f1': [base_f1],
    #                               'precision': [base_precision],
    #                               'recall': [base_recall]})
    

    # Compute and save the best fit model scores
    svm_best_results = model_scoring(X_test, y_test, svm_fit)
    svm_best_results.to_csv(os.path.join(results_to, "tables", "svm_score.csv"), index=False)

    # accuracy = svm_fit.score(X_test, y_test)
    # test_f1 = f1_score(y_test, svm_fit.predict(X_test), average='weighted')
    # precision = precision_score(y_test, svm_fit.predict(X_test), average='weighted')
    # recall = recall_score(y_test, svm_fit.predict(X_test), average='weighted')

    # results_table = pd.DataFrame({'accuracy': [accuracy],
    #                               'f1': [test_f1],
    #                               'precision': [precision],
    #                               'recall': [recall]})
    

    

    # Create and save confusion matrix using utility function
    final_svm_pred = svm_fit.predict(X_test)
    labels = ['Theft from Vehicle', 'Mischief', 'Break and Enter Residential/Other', 'Offence Against a Person']
    
    create_confusion_matrix(
        y_test=y_test,
        y_pred=final_svm_pred,
        labels=labels,
        title='Confusion Matrix - SVM',
        save_path=os.path.join(plot_to, "svm_confusion_matrix.png")
    )

    # Create and save classification report
    class_report = pd.DataFrame(classification_report(y_test, final_svm_pred, output_dict=True))
    class_report.to_csv(os.path.join(results_to, "tables", "svm_class_report.csv"), index=False)

if __name__ == '__main__':
    svm_eval()
