# plotting.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-09

import altair as alt
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

    chart = alt.Chart(df).mark_bar().encode(

        x = alt.X(x_col) if x_col != 'count()' else alt.X('count():Q'),
        y = alt.Y(y_col)
    ).properties(title  = title, width = width, height = height)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chart.save(save_path, scale_factor=2)