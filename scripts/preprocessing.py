# preprocessing.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01


import click
import os
import numpy as np
import pandas as pd
import pandera as pa
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer


@click.command()
@click.option('--raw-data', type=str, help="Path to raw crime dataset")
@click.option('--data-to', type=str, help="Path to save processed CSV")
@click.option('--preprocessor-to', type=str, help="Path to save preprocessor object")
@click.option('--seed', type=int, default=123, help="Random seed")


def preprocess(input_csv, output_csv):
    """
    This script loads the Vancouver crime dataset, performs preprocessing and 
    feature engineering, selects the top 4 crime types, validates the data, 
    splits into train/test sets, and saves the processed outputs along with 
    the fitted preprocessing pipeline.
    """
    
    np.random.seed(seed)

    # Load data
    df = pd.read_csv(raw_data)


    # Create a working copy
    df_processed = df.copy()

#   Fill missing HOUR and MINUTE values first (using mode)
    hour_mode = df_processed['HOUR'].mode()[0]
    minute_mode = df_processed['MINUTE'].mode()[0]

    df_processed['HOUR'] = df_processed['HOUR'].fillna(hour_mode).astype(int)
    df_processed['MINUTE'] = df_processed['MINUTE'].fillna(minute_mode).astype(int)

    # Create a proper datetime column
    df_processed['DATETIME'] = pd.to_datetime(
        df_processed['YEAR'].astype(str) + '-' +
        df_processed['MONTH'].astype(str) + '-' +
        df_processed['DAY'].astype(str) + ' ' +
        df_processed['HOUR'].astype(str) + ':' +
        df_processed['MINUTE'].astype(str)
    )
    # Extract day of week (0=Monday, 6=Sunday)
    df_processed['DAY_OF_WEEK'] = df_processed['DATETIME'].dt.dayofweek

    # Create weekend indicator
    df_processed['IS_WEEKEND'] = (df_processed['DAY_OF_WEEK'] >= 5).astype(int)
    

    # Time of day categories
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    df_processed['TIME_OF_DAY'] = df_processed['HOUR'].apply(categorize_time)

    # Season from month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df_processed['SEASON'] = df_processed['MONTH'].apply(get_season)

    # Cyclical encoding for time features
    df_processed['HOUR_SIN'] = np.sin(2 * np.pi * df_processed['HOUR'] / 24)
    df_processed['HOUR_COS'] = np.cos(2 * np.pi * df_processed['HOUR'] / 24)
    df_processed['MONTH_SIN'] = np.sin(2 * np.pi * df_processed['MONTH'] / 12)
    df_processed['MONTH_COS'] = np.cos(2 * np.pi * df_processed['MONTH'] / 12)

    # Rush hour indicator (morning and evening commute)
    df_processed['IS_RUSH_HOUR'] = df_processed['HOUR'].apply(
        lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
    )

    # Late night indicator
    df_processed['IS_LATE_NIGHT'] = df_processed['HOUR'].apply(
        lambda x: 1 if (x >= 22 or x <= 4) else 0
    )

    # Distance from downtown Vancouver (approximate coordinates)
    downtown_x = 491500
    downtown_y = 5459000
    df_processed['DIST_FROM_DOWNTOWN'] = np.sqrt(
        (df_processed['X'] - downtown_x)**2 +
        (df_processed['Y'] - downtown_y)**2
    )

    #print("Feature Engineering Complete")
    #print(f"Original columns: {len(df.columns)}")
    ##print(f"New columns: {len(df_processed.columns)}")
    #p#rint(f"Features added: {len(df_processed.columns) - len(df.columns)}")

    # Check distribution of crime types
    crime_counts = df_processed['TYPE'].value_counts()
    #print("Top 10 Crime Types:")
    #print(crime_counts.head(10))
    #print(f"\nTotal crime types in dataset: {len(crime_counts)}")

    # Selecting top 4 crime types
    n_classes = 4
    selected_crimes = crime_counts.head(n_classes).index.tolist()

    #print(f"\nSelected {n_classes} crime types for classification:")
    for i, crime in enumerate(selected_crimes, 1):
        pct = (crime_counts[crime] / len(df_processed)) * 100
        print(f"{i}. {crime}: {crime_counts[crime]:,} ({pct:.1f}%)")

    # Filter dataset
    df_model = df_processed[df_processed['TYPE'].isin(selected_crimes)].copy()

    #print(f"\nFiltered dataset size: {len(df_model):,} records")
    #print(f"This represents {len(df_model)/len(df_processed)*100:.1f}% of all crimes")

    numeric_cols = [
        'HOUR', 'DAY_OF_WEEK', 'MONTH', 'DAY',
        'IS_WEEKEND', 'IS_RUSH_HOUR', 'YEAR', 'IS_LATE_NIGHT',
        'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS',
        'DIST_FROM_DOWNTOWN', 'X', 'Y'
    ]

    categorical_cols = ['NEIGHBOURHOOD', 'TIME_OF_DAY', 'SEASON']



