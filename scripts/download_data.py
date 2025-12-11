# download_data.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01

"""
Data download script for Vancouver Crime Predictor.

This module provides functionality to download the Vancouver crime dataset from
Kaggle and save it locally as both CSV and ZIP formats for analysis.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-01
"""

import click
import kagglehub
import os
import zipfile
import pandas as pd
from src.data_utils import download_and_load_csv, save_csv_and_zip

@click.command()
@click.option('--dataset', required=True, help='Kaggle dataset identifier')
@click.option('--output-csv', required=True, help='Path to save the CSV file')
@click.option('--output-zip', required=True, help='Path to save the zipped file')

def download_data(dataset, output_csv, output_zip):
    """Download dataset from Kaggle and save locally as CSV and ZIP."""
    

    try:
        df = download_and_load_csv(dataset)
        save_csv_and_zip(df, output_csv, output_zip)
    except Exception as e:
        raise ValueError(f"File format issue: {e}")
    
    

if __name__ == '__main__':
    download_data()