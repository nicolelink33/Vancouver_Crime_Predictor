# test_plotting.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-10


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
