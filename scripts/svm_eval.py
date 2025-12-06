# svm_eval.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-05

import click
import pickle
import os
import altair as alt
import numpy as np
import pandas as pd

@click.command()
@click.option('--X_test_path', type=str, help="Path to X_test")
@click.option('--y_test_path', type=str, help="Path to y_test")
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plots will be written to")
def svm_eval(X_test_path, y_test_path, pipeline-from, results-to, plot-to):
    '''Evaluates the Vancouver Crime Predictor on the test data 
    and saves the evaluation results.'''

    # Read in the data and fitted svm model
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    with open(os.path.join(pipeline_from, "svm_final_best_fit.pickle"), 'rb') as f:
        svm_fit = pickle.load(f)
    
    # Compute accuracy
    accuracy = svm_fit.score(X_test, y_test)
    accuracy_table = pd.DataFrame({'accuracy': [accuracy]})
    accuracy_table.to_csv(os.path.join(results_to, "svm_score.csv"), index=False)

    # Create and save confusion matrix
    final_svm_pred = final_svm.predict(X_test)

    cm_svm = confusion_matrix(y_test, final_svm_pred)
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Theft from Vehicle', 'Mischief', 'Break and Enter Residential/Other', 'Offence Against a Person'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    plt.title(f'Confusion Matrix - SVM', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Crime Type', fontsize=12)
    plt.ylabel('Actual Crime Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.save(os.path.join(plot_to, "svm_confusion_matrix"), scale_factor=2.0)

    # Create and save classification report
    class_report = pd.DataFrame(classification_report(y_test, final_svm_pred, output_dict=True))
    class_report.to_csv(os.path.join(results_to, "svm_class_report.csv"), index=False)

if __name__ == '__main__':
    svm_eval()
