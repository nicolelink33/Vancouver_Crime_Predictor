"""
Plotting utilities for creating and saving visualizations.

This module provides reusable functions for generating and saving
bar plots using Altair. The functions handle creating directories
if they do not exist, and allow saving plots as HTML or PNG files.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-09
"""

import altair as alt
import pandas as pd
import os


def save_bar_plot(df, x_col, y_col, title, save_path, width = 700, height = 400):

    """
    Creates a bar plot using Altair and saves it to the specified path.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        Column name for the x-axis. If 'count()', the function will count records for each y category.
    y_col : str
        Column name for the y-axis.
    title : str
        Title of the plot.
    save_path : str
        File path to save the plot (PNG or HTML).
    width : int, optional
        Width of the plot in pixels (default is 700).
    height : int, optional
        Height of the plot in pixels (default is 400).

    Returns:
    -------
    None
        The plot is saved to the specified path. Nothing is returned.

    Raises:
    -------
    TypeError
        If df is not a DataFrame, or if x_col, y_col, title, or save_path are not strings.
    ValueError
        If df is empty, if column names are not found in DataFrame, or if save_path is empty.

    Examples:
    -------
    >>> save_bar_plot(
    >>> df=df,
    >>> x_col='HOUR:O',
    >>> y_col='count():Q',
    >>> title='Crimes by Hour of Day',
    >>> save_path=os.path.join(plot_to, "crimes_by_hour.png")
    )

    Notes:
    -----
    - If x_col is 'count()', the function counts the number of records grouped by y_col.
    - The directory for save_path will be created if it does not exist.
    - This function uses Altair to generate bar plots.
    """
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not isinstance(x_col, str):
        raise TypeError("x_col must be a string")
    if not isinstance(y_col, str):
        raise TypeError("y_col must be a string")
    if x_col != 'count()' and x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(save_path, str):
        raise TypeError("save_path must be a string")
    if not save_path or not save_path.strip():
        raise ValueError("save_path cannot be empty")

    chart = alt.Chart(df).mark_bar().encode(

        x = alt.X(x_col) if x_col != 'count()' else alt.X('count():Q'),
        y = alt.Y(y_col)
    ).properties(title  = title, width = width, height = height)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chart.save(save_path, scale_factor=2)