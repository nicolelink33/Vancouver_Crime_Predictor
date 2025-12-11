# test_data_utils.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-10


import os
import pandas as pd
import zipfile
import pytest
from unittest.mock import patch, MagicMock

from src.data_utils import download_and_load_csv, save_csv_and_zip


# ---------------------------------------------------------------------
# Test: download_and_load_csv
# ---------------------------------------------------------------------
def test_download_and_load_csv(tmp_path):
    """
    Test that download_and_load_csv returns a DataFrame and correctly
    reads a CSV file from a mocked kagglehub download path.
    """

    # Create a fake directory KaggleHub would "download"
    fake_kaggle_path = tmp_path / "fake_dataset"
    fake_kaggle_path.mkdir()

    # Create a fake CSV file inside it
    csv_file = fake_kaggle_path / "crime.csv"
    df_expected = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_expected.to_csv(csv_file, index=False)

    # Mock kagglehub.dataset_download to return our fake folder
    with patch("src.data_utils.kagglehub.dataset_download", return_value=str(fake_kaggle_path)):
        df_result = download_and_load_csv("username/dataset")

    # Assertions
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.equals(df_expected)


# ---------------------------------------------------------------------
# Test: save_csv_and_zip
# ---------------------------------------------------------------------
def test_save_csv_and_zip(tmp_path):
    """
    Test that save_csv_and_zip correctly writes a CSV file and then
    compresses it into a ZIP archive.
    """

    # Prepare test data
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})

    output_csv = tmp_path / "test.csv"
    output_zip = tmp_path / "test.zip"

    # Run function
    save_csv_and_zip(df, str(output_csv), str(output_zip))

    # Check CSV exists
    assert output_csv.exists()
    loaded = pd.read_csv(output_csv)
    assert loaded.equals(df)

    # Check ZIP exists
    assert output_zip.exists()

    # Check ZIP contains the CSV file
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        assert "test.csv" in zipf.namelist()
