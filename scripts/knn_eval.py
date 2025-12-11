# knn_eval.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-06

"""
K-Nearest Neighbors model evaluation script for Vancouver Crime Predictor.

This module evaluates the trained KNN model on test data, generating performance
metrics, confusion matrices, and classification reports.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-06
"""

import click
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.confusion_matrix_utils import create_confusion_matrix

@click.command()
@click.option('--x-test-path', type=str, default='data/processed/X_test.csv', help="Path to X_test CSV")
@click.option('--y-test-path', type=str, default='data/processed/y_test.csv', help="Path to y_test CSV")
@click.option('--model-path', type=str, default='results/models/knn_model.pickle', help="Path to trained KNN model")
@click.option('--plot-out', type=str, default='results/figures/knn_confusion_matrix.png', help="Path to save confusion matrix plot")
@click.option('--report-out', type=str, default='results/tables/knn_class_report.txt', help="Path to save classification report")
@click.option('--results-to', type=str, default='results/tables', help="Directory to save results table")
def knn_eval(x_test_path, y_test_path, model_path, plot_out, report_out, results_to):
    """
    Evaluate the trained KNN model on test data and save results.
    Generates accuracy score, confusion matrix, and classification report.
    Saves metrics table (accuracy, F1, precision, recall) as CSV.
    """
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
    os.makedirs(os.path.dirname(report_out), exist_ok=True)
    os.makedirs(results_to, exist_ok=True)
    
    # Load test data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # Load trained baseline KNN model
    baseline_path = model_path.replace('.pickle', '_baseline.pickle')
    with open(baseline_path, 'rb') as f:
        baseline_model = pickle.load(f)

    # Evaluate baseline model
    y_base_pred = baseline_model.predict(X_test)

    base_accuracy = baseline_model.score(X_test, y_test)
    base_f1 = f1_score(y_test, baseline_model.predict(X_test), average='weighted')
    base_precision = precision_score(y_test, baseline_model.predict(X_test), average='weighted')
    base_recall = recall_score(y_test, baseline_model.predict(X_test), average='weighted')

    results_table = pd.DataFrame({'accuracy': [base_accuracy],
                                  'f1': [base_f1],
                                  'precision': [base_precision],
                                  'recall': [base_recall]})
    results_table.to_csv(os.path.join(results_to, "tables", "knn_baseline_score.csv"), index=False)
    
    # Load trained KNN model
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)
    
    print("Evaluating KNN model on test data...")
    
    # Generate predictions
    y_pred = knn_model.predict(X_test)
    
    # Compute metrics
    test_accuracy = knn_model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Save metrics table as CSV (matching Nicole's SVM format)
    results_table = pd.DataFrame({
        'accuracy': [test_accuracy],
        'f1': [f1],
        'precision': [precision],
        'recall': [recall]
    })
    results_table.to_csv(os.path.join(results_to, "tables", "knn_score.csv"), index=False)
    
    print(f"Metrics table saved to {results_to}/knn_score.csv")
    
    # Calculate additional F1 scores for reporting
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Create classification report
    report = classification_report(y_test, y_pred)
    
    # Save classification report with metrics
    with open(report_out, 'w') as f:
        f.write("KNN Classification Report\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(report)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Classification report saved to {report_out}")
    
    # Get crime type labels
    crime_types = knn_model.classes_
    
    # Create confusion matrix using utility function
    create_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        labels=list(crime_types),
        title='Confusion Matrix - KNN Model',
        save_path=plot_out
    )
    
    print(f"Confusion matrix saved to {plot_out}")
    
    # Print summary
    print("\n" + "="*60)
    print("KNN MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*60)

if __name__ == '__main__':
    knn_eval()
