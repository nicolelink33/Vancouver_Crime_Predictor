# eda.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01

"""
Exploratory Data Analysis script for Vancouver Crime Predictor.

This module generates visualizations and summary statistics to understand patterns
in the Vancouver crime dataset, including temporal trends and spatial distributions.

Author: Nicole Link, Zain Nofal, Tirth Joshi
Date: 2025-12-01
"""
import click
import os
import pandas as pd
import altair as alt
import folium
from src.plotting import save_bar_plot


@click.command()
@click.option('--processed-training-data',type=str,required=True,help="Path to processed training CSV containing features")
@click.option('--plot-to',type=str,required=True,help="Directory to save the generated EDA plots.")
@click.option('--target-csv', type=str, required=True, help="Path to CSV file containing the target column TYPE")


def eda(processed_training_data,  target_csv, plot_to):
    """
    Generates exploratory data analysis visualizations for the crime dataset.
    Saves static PNG files for each plot.

    Expected columns:
    YEAR, MONTH, DAY, HOUR, NEIGHBOURHOOD, Latitude, Longitude, TYPE
    """
     

    os.makedirs(plot_to, exist_ok=True)

    # Load data
    df_features = pd.read_csv(processed_training_data)
    df_target = pd.read_csv(target_csv)

    # Merge target column like in notebook
    df = df_features.copy()
    df['TYPE'] = df_target['TYPE'].values

    alt.data_transformers.disable_max_rows()

    # Crime distribution plot
    save_bar_plot (
        df = df,
        x_col = 'count()',
        y_col = 'TYPE:N',
        title='Distribution of Crime Types',
        save_path=os.path.join(plot_to, "crime_type_distribution.png") 
    )
    # Crime Trend Over Years
    save_bar_plot(
    df=df,
    x_col='YEAR:O',
    y_col='count():Q',
    title='Crime Trend Over Years',
    save_path=os.path.join(plot_to, "crime_trend_years.png")    
    )

# Crimes by hour
    save_bar_plot(
    df=df,
    x_col='HOUR:O',
    y_col='count():Q',
    title='Crimes by Hour of Day',
    save_path=os.path.join(plot_to, "crimes_by_hour.png")
    )

# Top 10 Neighborhood with crimes
    neighbourhood_counts = (
        df["NEIGHBOURHOOD"]
        .value_counts()
        .reset_index(name='count')
        .rename(columns={"index": "NEIGHBOURHOOD"})
        .head(10)
    )

    save_bar_plot(
    df=neighbourhood_counts,
    x_col='count',
    y_col='NEIGHBOURHOOD:N',
    title='Top 10 Neighbourhoods by Crime Count',
    save_path=os.path.join(plot_to, "top10_neighbourhoods.png")
    )   


    #folium map viz
    # Center map around Vancouver
    crime_map = folium.Map(location=[49.2827, -123.1207], zoom_start=12)

    # Add markers (limit to first 2000 points to avoid clutter)
    for _, row in df.head(2000).iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.5,
            popup=f"{row['TYPE']} - {row['NEIGHBOURHOOD']}"
        ).add_to(crime_map)

    # Save map as HTML
    map_path = os.path.join(plot_to, "crime_map.html")
    crime_map.save(map_path)

    print(f"EDA completed. Plots and map saved in {plot_to}")

if __name__ == '__main__':
    eda()
    



