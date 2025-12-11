# log_reg_eval.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-06

"""
Logistic Regression model evaluation script for Vancouver Crime Predictor.

This module evaluates the trained Logistic Regression model on test data, generating
performance metrics, confusion matrices, and classification reports.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-06
"""

import click
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from src.confusion_matrix_utils import create_confusion_matrix

@click.command()
@click.option('--x-test-path', type=str, required=True,help="Path to X_test CSV file")
@click.option('--y-test-path', type=str, required=True,help="Path to y_test CSV file")
@click.option('--model-path', type=str, required=True,help="Path to location of trained Logistic Regression model pickle")
@click.option('--plot-out', type=str, required=True,help="File path for saving confusion matrix plot")
@click.option('--report-out', type=str, required=True,help="File path for saving classification report text")


def log_reg_eval(x_test_path, y_test_path, model_path, plot_out, report_out):
    """
    Evaluates a trained Logistic Regression model on test data.
    Saves a confusion matrix plot and classification report.
    """

    # Load data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Load baseline model
    with open(os.path.join(model_path, "logreg_baseline_fit.pickle"), "rb") as f:
        baseline_model = pickle.load(f)

    y_base_pred = baseline_model.predict(X_test)

    # Accuracy & report
    base_accuracy = baseline_model.score(X_test, y_test)
    base_f1 = f1_score(y_test, y_base_pred, average='weighted')
    base_precision = precision_score(y_test, y_base_pred, average='weighted')
    base_recall = recall_score(y_test, y_base_pred, average='weighted')

    results_table = pd.DataFrame({'accuracy': [base_accuracy],
                                  'f1': [base_f1],
                                  'precision': [base_precision],
                                  'recall': [base_recall]})
    results_table.to_csv(os.path.join(report_out, "logreg_baseline_score.csv"), index=False)
    
    
    # Load trained model
    with open(os.path.join(model_path, "log_reg_model.pickle"), "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy & report
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    report = classification_report(y_test, y_pred)


    # Save metrics to CSV
    metrics_df = pd.DataFrame({
    'accuracy': [accuracy],
    'f1': [f1],
    'precision': [precision],
    'recall': [recall]})
    metrics_df.to_csv(os.path.join(report_out, "logreg_score.csv"), index=False)

    with open(os.path.join(report_out, "log_reg_class_report.txt"), "w") as f:
        f.write("Logistic Regression Classification Report\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(report)
        
    # Create confusion matrix using utility function
    create_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        labels=list(model.classes_),
        title='Confusion Matrix - Logistic Regression',
        save_path=plot_out
    )

    print("\nEvaluation complete.")
    print("Confusion matrix saved to:", plot_out)
    print("Classification report saved to:", report_out, "/log_reg_class_report.txt")


if __name__ == '__main__':
    log_reg_eval()