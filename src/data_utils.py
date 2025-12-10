# data_utils.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-09

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

    path = kagglehub.dataset_download(dataset)
    df = pd.read_csv(os.path.join(path, csv_filename))
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

    Examples:
    --------
    >>> save_csv_and_zip(df, 'data/crime.csv', 'data/crime.zip')

    Notes:
    -----
    This function ensures both CSV and compressed ZIP versions of the data are saved for portability.
    """
    df.to_csv(output_csv, index=False)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname=os.path.basename(output_csv))
