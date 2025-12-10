"""
svm_small_data_test.py
Function to test Linear SVM model performance on a subset of the features

This function takes in training data and a created preprocessor, and creates a Linear SVM model. 
It also creates a subset of the training features, eliminating some of the features created in feature engineering. 
It then performs hyperparameter optimization to find the best fit model, fits this optimized model, and returns the model as a pickle object. 
It also creates a plot for the hyperparameter optimization and saves it, and also saves the results from cross validation. 

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-10
"""

import click
import pickle
import os
import altair as alt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

@click.command()
@click.option('--X_train_path', type=str, help="Path to X_train")
@click.option('--y_train_path', type=str, help="Path to y_train")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)
def svm_small_test(X_train_path, y_train_path, pipeline-to, plot-to, seed):
    # Read in train data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # Create subsamples of X_train and y_train to speed up optimization
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train,
        train_size=15000,
        stratify=y_train,
        random_state=seed
    )

    # Create column lists: to scale and to drop
    small_numeric = ['HOUR',
        'DAY_OF_WEEK',
        'MONTH',
        'DAY',
        'X',
        'Y',
        'IS_WEEKEND', 'IS_RUSH_HOUR',
        'IS_LATE_NIGHT', 'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS',
        'DIST_FROM_DOWNTOWN']

    drop_feats = [
        'NEIGHBOURHOOD_Central Business District',
        'NEIGHBOURHOOD_Dunbar-Southlands',
        'NEIGHBOURHOOD_Fairview',
        'NEIGHBOURHOOD_Grandview-Woodland',
        'NEIGHBOURHOOD_Hastings-Sunrise',
        'NEIGHBOURHOOD_Kensington-Cedar Cottage',
        'NEIGHBOURHOOD_Kerrisdale',
        'NEIGHBOURHOOD_Killarney',
        'NEIGHBOURHOOD_Kitsilano',
        'NEIGHBOURHOOD_Marpole',
        'NEIGHBOURHOOD_Mount Pleasant',
        'NEIGHBOURHOOD_Musqueam',
        'NEIGHBOURHOOD_Oakridge',
        'NEIGHBOURHOOD_Renfrew-Collingwood',
        'NEIGHBOURHOOD_Riley Park',
        'NEIGHBOURHOOD_Shaughnessy',
        'NEIGHBOURHOOD_South Cambie',
        'NEIGHBOURHOOD_Stanley Park',
        'NEIGHBOURHOOD_Strathcona',
        'NEIGHBOURHOOD_Sunset',
        'NEIGHBOURHOOD_Victoria-Fraserview',
        'NEIGHBOURHOOD_West End',
        'NEIGHBOURHOOD_West Point Grey',
        'TIME_OF_DAY_Evening',
        'TIME_OF_DAY_Morning',
        'TIME_OF_DAY_Night',
        'SEASON_Spring',
        'SEASON_Summer',
        'SEASON_Winter'
    ]

    # Create column transformer and pipeline
    small_svm_transf = make_column_transformer(
        (StandardScaler(), small_numeric),
        ("drop", drop_feats)
    )
    small_svm_pipe = make_pipeline(
        small_svm_transf,
        LinearSVC()
    )
    
    # Perform hyperparameter optimization to find best C
    param_grid = {
        "linearsvc__C": loguniform(1e-3, 1e2)
    }

    small_svm_random_search = RandomizedSearchCV(small_svm_pipe, 
        param_distributions = param_grid, 
        n_iter=10, 
        cv=5,
        n_jobs=-1, 
        return_train_score=True)
    small_svm_random_search.fit(X_train_sample, y_train_sample)

    # Plot results of random search with fewer features
    small_svm_results = pd.DataFrame(small_svm_random_search.cv_results_
        ).set_index("rank_test_score"
        ).sort_index()
    
    results_plot = alt.Chart(small_svm_results, 
        title=alt.Title(
            text="Finding Optimal C Value for Linear SVM")
        ).mark_circle(size=70
        ).encode(
            x=alt.X('param_linearsvc__C').title("C Value"),
            y=alt.Y('mean_test_score').scale(zero=False
                    ).title("Cross-Validation Accuracy")
        )
    
    results_plot.save(os.path.join(plot_to, "svm_few_features_results"), scale_factor=2.0)

if __name__ == '__main__':
    svm_small_test()