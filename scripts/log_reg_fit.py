# log_reg_fit.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-06

"""
Logistic Regression model training script for Vancouver Crime Predictor.

This module trains a Logistic Regression classifier to predict crime types in Vancouver,
including baseline and optimized models.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-06
"""

import click
import pandas as pd
import pickle
import os
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


@click.command()
@click.option('--x-train-path', type=str, required=True,help="Path to X_train CSV file")
@click.option('--y-train-path', type=str, required=True,help="Path to y_train CSV file")
@click.option('--model-out', type=str, required=True,help="File path where the trained model pickle will be saved")
@click.option('--params-out', type=str, required=True,help="File path where the best hyperparameters JSON will be saved")
@click.option('--seed', type=int, default=522,help="Random seed for reproducibility")

def log_reg_fit(x_train_path, y_train_path, model_out, params_out, seed):
    """
    Fits and tunes a Logistic Regression model using GridSearchCV.
    Saves the best model and hyperparameters.
    """

    # Load training data
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Convert to Series

    # Numerical and categorical columns
    numerical_cols = [
        'HOUR', 'DAY_OF_WEEK', 'MONTH', 'DAY',
        'IS_WEEKEND', 'IS_RUSH_HOUR', 'IS_LATE_NIGHT',
        'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS'
    ]
    categorical_cols = [col for col in X_train.columns if col not in numerical_cols]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ]
    )

    # Create and fit baseline model, save the model
    base_pipe = make_pipeline(
        preprocessor,
        LogisticRegression()
    )
    base_pipe.fit(X_train, y_train)

    with open(os.path.join(model_out, "logreg_baseline_fit.pickle"), 'wb') as f:
        pickle.dump(base_pipe, f)

    # Logistic Regression Pipeline
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            multi_class='multinomial',
            solver='saga',
            max_iter=200,
            random_state=seed
        ))
    ])

    # Hyperparameter grid
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    }

    grid = GridSearchCV(
        logreg_pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit
    grid.fit(X_train, y_train)

    # Save best model
    with open(os.path.join(model_out, "log_reg_model.pickle"), "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    # Save best hyperparameters
    with open(os.path.join(params_out, "log_reg_params.json"), "w") as f:
        json.dump(grid.best_params_, f, indent=4)

    print("\nBest Logistic Regression model saved to:", model_out, "log_reg_model.pickle")
    print("Best hyperparameters:", grid.best_params_)


if __name__ == '__main__':
    log_reg_fit()
