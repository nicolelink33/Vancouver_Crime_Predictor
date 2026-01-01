"""
Unit tests for data utility functions: download_and_load_csv and save_csv_and_zip.

This module provides tests to verify the correct behavior of functions in
src.data_utils, including CSV downloading/loading from Kaggle and saving
DataFrames to CSV and ZIP formats.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-10
"""


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


# ---------------------------------------------------------------------
# Error Handling Tests: download_and_load_csv
# ---------------------------------------------------------------------
def test_download_and_load_csv_invalid_dataset_type():
    """Test that invalid dataset type raises TypeError"""
    with pytest.raises(TypeError, match="dataset must be a string"):
        download_and_load_csv(123)


def test_download_and_load_csv_empty_dataset():
    """Test that empty dataset raises ValueError"""
    with pytest.raises(ValueError, match="dataset cannot be empty"):
        download_and_load_csv("")


def test_download_and_load_csv_whitespace_dataset():
    """Test that whitespace-only dataset raises ValueError"""
    with pytest.raises(ValueError, match="dataset cannot be empty"):
        download_and_load_csv("   ")


def test_download_and_load_csv_invalid_csv_filename_type():
    """Test that invalid csv_filename type raises TypeError"""
    with pytest.raises(TypeError, match="csv_filename must be a string"):
        download_and_load_csv("valid/dataset", 123)


def test_download_and_load_csv_empty_csv_filename():
    """Test that empty csv_filename raises ValueError"""
    with pytest.raises(ValueError, match="csv_filename cannot be empty"):
        download_and_load_csv("valid/dataset", "")


def test_download_and_load_csv_missing_file(tmp_path):
    """Test that missing CSV file raises FileNotFoundError"""
    fake_kaggle_path = tmp_path / "fake_dataset"
    fake_kaggle_path.mkdir()
    # Don't create the CSV file
    
    with patch("src.data_utils.kagglehub.dataset_download", return_value=str(fake_kaggle_path)):
        with pytest.raises(FileNotFoundError, match="CSV file .* not found"):
            download_and_load_csv("username/dataset", "nonexistent.csv")


# ---------------------------------------------------------------------
# Error Handling Tests: save_csv_and_zip
# ---------------------------------------------------------------------
def test_save_csv_and_zip_invalid_df_type(tmp_path):
    """Test that invalid DataFrame type raises TypeError"""
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        save_csv_and_zip("not a dataframe", str(tmp_path / "out.csv"), str(tmp_path / "out.zip"))


def test_save_csv_and_zip_empty_df(tmp_path):
    """Test that empty DataFrame raises ValueError"""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        save_csv_and_zip(empty_df, str(tmp_path / "out.csv"), str(tmp_path / "out.zip"))


def test_save_csv_and_zip_invalid_output_csv_type(tmp_path):
    """Test that invalid output_csv type raises TypeError"""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError, match="output_csv must be a string"):
        save_csv_and_zip(df, 123, str(tmp_path / "out.zip"))


def test_save_csv_and_zip_empty_output_csv(tmp_path):
    """Test that empty output_csv raises ValueError"""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="output_csv cannot be empty"):
        save_csv_and_zip(df, "", str(tmp_path / "out.zip"))


def test_save_csv_and_zip_invalid_output_zip_type(tmp_path):
    """Test that invalid output_zip type raises TypeError"""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError, match="output_zip must be a string"):
        save_csv_and_zip(df, str(tmp_path / "out.csv"), 123)


def test_save_csv_and_zip_empty_output_zip(tmp_path):
    """Test that empty output_zip raises ValueError"""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="output_zip cannot be empty"):
        save_csv_and_zip(df, str(tmp_path / "out.csv"), "")
