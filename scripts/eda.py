# eda.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-05


import click
import os
import pandas as pd
import altair as alt
import folium


@click.command()
@click.option('--processed-training-data',type=str,required=True,help="Path to processed training CSV containing features + TYPE column.")
@click.option('--plot-to',type=str,required=True,help="Directory to save the generated EDA plots.")

def eda(processed_training_data, plot_to):
    """
    Generates exploratory data analysis visualizations for the crime dataset.
    Saves static PNG files for each plot.

    Expected columns:
    YEAR, MONTH, DAY, HOUR, NEIGHBOURHOOD, Latitude, Longitude, TYPE
    """
     

    os.makedirs(plot_to, exist_ok=True)

    # Load data
    df = pd.read_csv(processed_training_data)


    alt.data_transformers.disable_max_rows()

    # Crime distribution plot

    type_plot = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y('TYPE:N', sort='-x', title='Crime Type'),
            x=alt.X('count():Q', title='Number of Crimes')
        )
        .properties(
            title='Distribution of Crime Types',
            width=700,
            height=400
        )
    )
    type_plot.save(os.path.join(plot_to, "crime_type_distribution.png"), scale_factor=2)

    year_plot = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X('YEAR:O', title='Year'),
            y=alt.Y('count():Q', title='Number of Crimes')
        )
        .properties(
            title='Crime Trend Over Years',
            width=700,
            height=400
        )
    )

    year_plot.save(os.path.join(plot_to, "crime_trend_years.png"), scale_factor=2)


    hour_plot = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('HOUR:O', title='Hour of Day'),
            y=alt.Y('count():Q', title='Number of Crimes')
        )
        .properties(
            title='Crimes by Hour of Day',
            width=700,
            height=400
        )
    )
    hour_plot.save(os.path.join(plot_to, "crimes_by_hour.png"), scale_factor=2)


    neighbourhood_counts = (
        df["NEIGHBOURHOOD"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "NEIGHBOURHOOD", "NEIGHBOURHOOD": "count"})
        .head(10)
    )

    neighbourhood_plot = (
        alt.Chart(neighbourhood_counts)
        .mark_bar()
        .encode(
            y=alt.Y('NEIGHBOURHOOD:N', sort='-x', title='Neighbourhood'),
            x=alt.X('count:Q', title='Number of Crimes')
        )
        .properties(
            title='Top 10 Neighbourhoods by Crime Count',
            width=700,
            height=400
        )
    )
    neighbourhood_plot.save(os.path.join(plot_to, "top10_neighbourhoods.png"), scale_factor=2)



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
            popup=f"{row[target]} - {row['NEIGHBOURHOOD']}"
        ).add_to(crime_map)

    # Save map as HTML
    map_path = os.path.join(output_dir, "crime_map.html")
    crime_map.save(map_path)

    print(f"EDA completed. Plots and map saved in {output_dir}")

if __name__ == '__main__':
    main()
    



