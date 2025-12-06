# knn_eval.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-06

import click
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

@click.command()
@click.option('--X-test-path', type=str, required=True, help="Path to X_test CSV")
@click.option('--y-test-path', type=str, required=True, help="Path to y_test CSV")
@click.option('--pipeline-from', type=str, required=True, help="Directory containing trained model")
@click.option('--results-to', type=str, required=True, help="Directory to save result tables")
@click.option('--plot-to', type=str, required=True, help="Directory to save plots")
def knn_eval(x_test_path, y_test_path, pipeline_from, results_to, plot_to):
    """
    Evaluate the trained KNN model on test data and save results.
    Generates accuracy score, confusion matrix, and classification report.
    """
    
    # Create output directories if they don't exist
    os.makedirs(results_to, exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)
    
    # Load test data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # Load trained KNN model
    model_path = os.path.join(pipeline_from, "knn_final_fit.pickle")
    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)
    
    print("Evaluating KNN model on test data...")
    
    # Compute test accuracy
    test_accuracy = knn_model.score(X_test, y_test)
    
    # Save accuracy
    accuracy_df = pd.DataFrame({'accuracy': [test_accuracy]})
    accuracy_df.to_csv(os.path.join(results_to, "knn_test_score.csv"), index=False)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    y_pred = knn_model.predict(X_test)
    
    # Create classification report
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict).transpose()
    class_report_df.to_csv(os.path.join(results_to, "knn_classification_report.csv"))
    
    print("Classification report saved.")
    
    # Get crime type labels
    crime_types = sorted(y_test.unique())
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=crime_types)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=crime_types)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    plt.title('Confusion Matrix - KNN Model', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Crime Type', fontsize=12)
    plt.ylabel('Actual Crime Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "knn_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved.")
    
    # Print summary
    print("\n" + "="*60)
    print("KNN MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"\nResults saved to: {results_to}")
    print(f"Plots saved to: {plot_to}")
    print("="*60)

if __name__ == '__main__':
    knn_eval()
