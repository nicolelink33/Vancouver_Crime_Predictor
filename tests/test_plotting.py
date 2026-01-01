"""
Unit tests for plotting utility functions: save_bar_plot.

This module provides tests to verify the correct behavior of functions in
src.plotting, ensuring that bar plots are correctly generated, saved, and
that directories are created automatically.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-10
"""


import os
import pandas as pd
import pytest
from src.plotting import save_bar_plot

@pytest.fixture
def sample_df():
    """Fixture: Small sample DataFrame for plotting tests."""
    return pd.DataFrame({
        "category": ["A", "B", "C"],
        "value": [10, 20, 15]
    })


def test_save_bar_plot_creates_html_file(tmp_path, sample_df):
    """
    Test that save_bar_plot creates an HTML file and
    that the directory is automatically created.
    """
    output_dir = tmp_path / "plots"
    output_path = output_dir / "test_plot.html"

    save_bar_plot(
        df=sample_df,
        x_col="category",
        y_col="value",
        title="Test Plot",
        save_path=str(output_path)
    )

    # Assert file exists
    assert output_path.exists(), "HTML plot file was not created."


def test_save_bar_plot_creates_directory_if_missing(tmp_path, sample_df):
    """
    Test that save_bar_plot automatically creates directories
    for the output path.
    """
    nested_dir = tmp_path / "nested" / "plots"
    output_path = nested_dir / "test_plot.html"

    save_bar_plot(
        df=sample_df,
        x_col="category",
        y_col="value",
        title="Test Plot",
        save_path=str(output_path)
    )

    # Directory should have been created
    assert nested_dir.exists(), "Output directory was not created."

    # File exists
    assert output_path.exists(), "Plot HTML file not saved in nested directory."


def test_save_bar_plot_count_special_case(tmp_path, sample_df):
    """
    Test the 'count()' special handling for x_col.
    (This ensures that x_col='count()' does not cause errors.)
    """
    output_path = tmp_path / "count_test.html"

    # This should not raise an error
    save_bar_plot(
        df=sample_df,
        x_col="count()",
        y_col="category",
        title="Count Test",
        save_path=str(output_path)
    )

    assert output_path.exists(), "HTML plot with count() was not created."


# ---------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------
def test_save_bar_plot_invalid_df_type(tmp_path):
    """Test that invalid DataFrame type raises TypeError"""
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        save_bar_plot("not a df", "x", "y", "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_empty_df(tmp_path):
    """Test that empty DataFrame raises ValueError"""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        save_bar_plot(empty_df, "x", "y", "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_x_column(tmp_path, sample_df):
    """Test that invalid x column name raises ValueError"""
    with pytest.raises(ValueError, match="Column .* not found"):
        save_bar_plot(sample_df, "nonexistent", "value", "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_y_column(tmp_path, sample_df):
    """Test that invalid y column name raises ValueError"""
    with pytest.raises(ValueError, match="Column .* not found"):
        save_bar_plot(sample_df, "category", "nonexistent", "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_x_col_type(tmp_path, sample_df):
    """Test that invalid x_col type raises TypeError"""
    with pytest.raises(TypeError, match="x_col must be a string"):
        save_bar_plot(sample_df, 123, "value", "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_y_col_type(tmp_path, sample_df):
    """Test that invalid y_col type raises TypeError"""
    with pytest.raises(TypeError, match="y_col must be a string"):
        save_bar_plot(sample_df, "category", 123, "Title", str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_title_type(tmp_path, sample_df):
    """Test that invalid title type raises TypeError"""
    with pytest.raises(TypeError, match="title must be a string"):
        save_bar_plot(sample_df, "category", "value", 123, str(tmp_path / "plot.html"))


def test_save_bar_plot_invalid_save_path_type(tmp_path, sample_df):
    """Test that invalid save_path type raises TypeError"""
    with pytest.raises(TypeError, match="save_path must be a string"):
        save_bar_plot(sample_df, "category", "value", "Title", 123)


def test_save_bar_plot_empty_save_path(tmp_path, sample_df):
    """Test that empty save_path raises ValueError"""
    with pytest.raises(ValueError, match="save_path cannot be empty"):
        save_bar_plot(sample_df, "category", "value", "Title", "")
