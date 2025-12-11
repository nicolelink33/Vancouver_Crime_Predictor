# knn_training.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-06

"""
K-Nearest Neighbors model training script for Vancouver Crime Predictor.

This module trains and optimizes a KNN classifier to predict crime types in Vancouver,
including hyperparameter tuning and model persistence.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-06
"""

import click
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

@click.command()
@click.option('--x-train-path', type=str, default='data/processed/X_train.csv', help="Path to X_train CSV")
@click.option('--y-train-path', type=str, default='data/processed/y_train.csv', help="Path to y_train CSV")
@click.option('--model-out', type=str, default='models/knn_model.pickle', help="Path to save trained KNN model")
@click.option('--plot-out', type=str, default='results/knn_k_optimization.png', help="Path to save optimization plot")
@click.option('--seed', type=int, default=522, help="Random seed for reproducibility")
def knn_fit(x_train_path, y_train_path, model_out, plot_out, seed):
    """
    Train and optimize KNN classifier for Vancouver crime type prediction.
    Tests k values from 5 to 100 and saves the best model.
    """
    
    np.random.seed(seed)
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)
    
    # Load data
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    
    # Determine numeric and categorical column indices
    num_numeric = 15  # Based on preprocessing.py numeric_cols
    numeric_indices = list(range(num_numeric))
    onehot_indices = list(range(num_numeric, X_train.shape[1]))
    
    # Create column transformer for scaling
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_indices),
            ('cat', 'passthrough', onehot_indices)
        ]
    )
    
    # Baseline model with k=5
    print("Training baseline KNN model (k=5)...")
    baseline_pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    
    # Save baseline model
    baseline_path = model_out.replace('.pickle', '_baseline.pickle')
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_pipeline, f)
    
    print(f"Baseline model saved to {baseline_path}")
    
    # Create subsample for faster k-value optimization
    print("Creating subsample for k-value optimization...")
    sample_size = 15000
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=seed
    )
    
    print(f"Using {len(X_train_sample):,} samples for k-value search")
    
    # K-value optimization
    print("Optimizing k value...")
    k_values = list(range(5, 101, 5))
    scores = []
    
    for k in k_values:
        pipeline = Pipeline([
            ('preprocessor', column_transformer),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        
        cv_score = cross_val_score(
            pipeline, X_train_sample, y_train_sample,
            cv=3, scoring='accuracy', n_jobs=-1
        ).mean()
        
        scores.append(cv_score)
        print(f"  k={k}: CV accuracy = {cv_score:.4f}")
    
    # Find best k
    best_k = k_values[np.argmax(scores)]
    best_score = max(scores)
    
    print(f"\nBest k: {best_k} (CV accuracy: {best_score:.4f})")
    
    # Create optimization plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'o-', linewidth=2, markersize=8, color='#4A90E2')
    plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
    plt.xlabel('k (number of neighbors)', fontsize=12)
    plt.ylabel('Cross-validation accuracy', fontsize=12)
    plt.title('Finding the Optimal k Value for KNN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimization plot saved to {plot_out}")
    
    # Train final model with best k on full training data
    print(f"\nTraining final KNN model with k={best_k} on full training data...")
    final_pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('knn', KNeighborsClassifier(n_neighbors=best_k))
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    # Save final model
    with open(model_out, 'wb') as f:
        pickle.dump(final_pipeline, f)
    
    print(f"\nFinal KNN model saved to {model_out}")
    print(f"Best k value: {best_k}")
    print(f"Optimization complete!")

if __name__ == '__main__':
    knn_fit()
