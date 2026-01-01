"""
Data utility functions for downloading, loading, and saving datasets.

This module provides functions to:
- Download a dataset from Kaggle using kagglehub and load it as a pandas DataFrame.
- Save a DataFrame to a CSV file and also compress it into a ZIP archive.

These functions are designed to be reusable across different projects where
Kaggle datasets are used or where saving data in both CSV and ZIP format is
desired for portability.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-09
"""

import os
import pandas as pd
import zipfile
import kagglehub

def download_and_load_csv(dataset: str, csv_filename: str = "crime.csv") -> pd.DataFrame:
    """
    Downloads a dataset from Kaggle using kagglehub, loads a CSV file from it, and returns a DataFrame.

    Parameters:
    ----------
    dataset : str
        Kaggle dataset identifier (e.g., 'username/dataset-name')
    csv_filename : str, default 'crime.csv'
        Name of the CSV file within the dataset to load.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the loaded CSV data.

    Raises:
    -------
    TypeError
        If dataset or csv_filename are not strings.
    ValueError
        If dataset or csv_filename are empty strings.
    FileNotFoundError
        If the CSV file does not exist in the downloaded dataset.
    Exception
        If there is an issue reading the CSV file.

    Examples:
    --------
    >>> df = download_and_load_csv('username/crime-dataset')
    >>> df.head()

    Notes:
    -----
    This function uses kagglehub to download the dataset. The CSV file must exist in the dataset directory.
    """
    
    # Input validation
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not dataset or not dataset.strip():
        raise ValueError("dataset cannot be empty")
    if not isinstance(csv_filename, str):
        raise TypeError("csv_filename must be a string")
    if not csv_filename or not csv_filename.strip():
        raise ValueError("csv_filename cannot be empty")

    path = kagglehub.dataset_download(dataset)
    csv_path = os.path.join(path, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_filename}' not found in dataset")
    
    df = pd.read_csv(csv_path)
    return df

def save_csv_and_zip(df: pd.DataFrame, output_csv: str, output_zip: str):
    """
    Saves a DataFrame to a CSV file and writes it into a ZIP archive.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    output_csv : str
        Path to save the CSV file.
    output_zip : str
        Path to save the ZIP file.

    Returns:
    -------
    None

    Raises:
    -------
    TypeError
        If df is not a DataFrame, or if output_csv or output_zip are not strings.
    ValueError
        If df is empty, or if output_csv or output_zip are empty strings.

    Examples:
    --------
    >>> save_csv_and_zip(df, 'data/crime.csv', 'data/crime.zip')

    Notes:
    -----
    This function ensures both CSV and compressed ZIP versions of the data are saved for portability.
    """
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not isinstance(output_csv, str):
        raise TypeError("output_csv must be a string")
    if not output_csv or not output_csv.strip():
        raise ValueError("output_csv cannot be empty")
    if not isinstance(output_zip, str):
        raise TypeError("output_zip must be a string")
    if not output_zip or not output_zip.strip():
        raise ValueError("output_zip cannot be empty")
    
    df.to_csv(output_csv, index=False)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname=os.path.basename(output_csv))
