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
from src.confusion_matrix_utils import create_confusion_matrix
from sklearn.metrics import classification_report
from src.model_scoring import model_scoring


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

    baseline_results = model_scoring(X_test, y_test, baseline_model)
    baseline_results.to_csv(
    os.path.join(report_out, "logreg_baseline_score.csv"),
    index=False
)
    
    
    # Load trained model
    with open(os.path.join(model_path, "log_reg_model.pickle"), "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    model_results = model_scoring(X_test, y_test, model)
    model_results.to_csv(
    os.path.join(report_out, "logreg_score.csv"),
    index=False
    )
    report = classification_report(y_test, y_pred)

    with open(os.path.join(report_out, "log_reg_class_report.txt"), "w") as f:
        f.write("Logistic Regression Classification Report\n")
        f.write(f"Accuracy: {model_results['accuracy'].iloc[0]:.4f}\n")
        f.write(f"Precision: {model_results['precision'].iloc[0]:.4f}\n")
        f.write(f"Recall: {model_results['recall'].iloc[0]:.4f}\n\n")
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