# evaluate_svm.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01

# passthrough_feats = ['NEIGHBOURHOOD_Central Business District',
#  'NEIGHBOURHOOD_Dunbar-Southlands',
#  'NEIGHBOURHOOD_Fairview',
#  'NEIGHBOURHOOD_Grandview-Woodland',
#  'NEIGHBOURHOOD_Hastings-Sunrise',
#  'NEIGHBOURHOOD_Kensington-Cedar Cottage',
#  'NEIGHBOURHOOD_Kerrisdale',
#  'NEIGHBOURHOOD_Killarney',
#  'NEIGHBOURHOOD_Kitsilano',
#  'NEIGHBOURHOOD_Marpole',
#  'NEIGHBOURHOOD_Mount Pleasant',
#  'NEIGHBOURHOOD_Musqueam',
#  'NEIGHBOURHOOD_Oakridge',
#  'NEIGHBOURHOOD_Renfrew-Collingwood',
#  'NEIGHBOURHOOD_Riley Park',
#  'NEIGHBOURHOOD_Shaughnessy',
#  'NEIGHBOURHOOD_South Cambie',
#  'NEIGHBOURHOOD_Stanley Park',
#  'NEIGHBOURHOOD_Strathcona',
#  'NEIGHBOURHOOD_Sunset',
#  'NEIGHBOURHOOD_Victoria-Fraserview',
#  'NEIGHBOURHOOD_West End',
#  'NEIGHBOURHOOD_West Point Grey',
#  'TIME_OF_DAY_Evening',
#  'TIME_OF_DAY_Morning',
#  'TIME_OF_DAY_Night',
#  'SEASON_Spring',
#  'SEASON_Summer',
#  'SEASON_Winter']

# numeric_cols = [
#     'HOUR', 'DAY_OF_WEEK', 'MONTH', 'DAY',
#     'IS_WEEKEND', 'IS_RUSH_HOUR', 'YEAR', 'IS_LATE_NIGHT',
#     'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS',
#     'DIST_FROM_DOWNTOWN', 'X', 'Y'
# ]

# categorical_cols = ['NEIGHBOURHOOD', 'TIME_OF_DAY', 'SEASON']


import click
import pickle
import os
import altair as alt
#import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

@click.command()
@click.option('--X_train', type=str, help="Path to X_train")
@click.option('--y_train', type=str, help="Path to y_train")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)

def svm_fitting(X_train, y_train, preprocessor, pipeline-to, plot-to, seed):
    # Create and fit baseline SVM model, save the model
    svm_base_pipe = make_pipeline(
        preprocessor,
        LinearSVC(C=1)
    )
    svm_base_pipe.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, "svm_baseline_fit.pickle"), 'wb') as f:
        pickle.dump(svm_base_pipe, f)
    # svm_base_pred = svm_base_pipe.predict(X_test)
    # svm_base_acc = accuracy_score(y_test, svm_base_pred)

    # Create subsamples of X_train and y_train to speed up optimization
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train,
        train_size=15000,
        stratify=y_train,
        random_state=seed
    )

    # Hyperparameter optimization for C
    svm_pipe = make_pipeline(
        preprocessor,
        LinearSVC()
    )
    param_grid = {
        "linearsvc__C": [0.001, 0.01, 0.5, 0.1, 1, 10, 50, 100]
    }
    svm_grid = GridSearchCV(
        svm_pipe,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        return_train_score=True
    )
    svm_grid.fit(X_train_sample, y_train_sample)

    # Save grid search fitted model
    with open (os.path.join(pipeline_to, "svm_initial_grid_fit.pickle"), 'wb') as f:
        pickle.dump(svm_grid, f)
    
    svm_grid_results = pd.DataFrame(svm_grid.cv_results_
        ).set_index("rank_test_score"
        ).sort_index()

    # Plot how accuracy changes with C
    svm_grid_results_plot = alt.Chart(svm_grid_results, 
        title=alt.Title(
            text="Finding Optimal C Value for Linear SVM by Grid Search")
            ).mark_circle(size=70
            ).encode(
                x=alt.X('param_linearsvc__C').title("C Value"),
                y=alt.Y('mean_test_score').scale(zero=False
                ).title("Cross-Validation Accuracy")
    )

    # Save the plot
    svm_grid_results_plot.save(os.path.join(plot_to, "svm_initial_grid_fit"), scale_factor=2.0)

    # Repeat hyperparameter optimization in the best range with RandomizedSearchCV
    param_dist = {
        "linearsvc__C": loguniform(1e-3, 1)
    }

    svm_random_search = RandomizedSearchCV(svm_pipe, 
        param_distributions = param_dist, 
        n_iter=10, 
        cv=5,
        n_jobs=-1, 
        return_train_score=True)
    svm_random_search.fit(X_train_sample, y_train_sample)

    # Save the random search model
    with open (os.path.join(pipeline_to, "svm_final_random_fit.pickle"), 'wb') as f:
        pickle.dump(svm_random_search, f)
    
    # Plot optimization results and save the plot
    svm_results = pd.DataFrame(svm_random_search.cv_results_
        ).set_index("rank_test_score"
        ).sort_index()
    svm_results

    svm_random_results_plot = alt.Chart(svm_results, 
        title=alt.Title(
            text="Finding Optimal C Value for Linear SVM")
        ).mark_circle(size=70
        ).encode(
            x=alt.X('param_linearsvc__C').title("C Value"),
            y=alt.Y('mean_test_score').scale(zero=False
            ).title("Cross-Validation Accuracy")
    )
    svm_random_results_plot.save(os.path.join(plot_to, "svm_final_random_fit"), scale_factor=2.0)

if __name__ == '__main__':
    svm_fitting()