# data_validation.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01

import click
import pandas as pd
import pandera as pa
import json
import logging

@click.command()
@click.option('--input-csv', required=True, help='Path to input CSV file')
@click.option('--output-csv', required=True, help='Path to save cleaned + validated CSV')

def data_validation(input_csv, output_csv):
    """Clean, validate, and save crime data."""

    logging.basicConfig(
        filename="validation_errors.log",
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    df = pd.read_csv(input_csv)

    crime_types = [
        'Other Theft', 
        'Break and Enter Residential/Other', 
        'Mischief',
        'Break and Enter Commercial', 
        'Offence Against a Person',
        'Theft from Vehicle',
        'Vehicle Collision or Pedestrian Struck (with Injury)',
        'Vehicle Collision or Pedestrian Struck (with Fatality)',
        'Theft of Vehicle', 
        'Homicide', 
        'Theft of Bicycle'
    ]

    clean_df = df.query("HUNDRED_BLOCK != 'OFFSET TO PROTECT PRIVACY'")


    dupes = clean_df[clean_df.duplicated()]
    if not dupes.empty:
        logging.warning(f"{len(dupes)} duplicate row(s) found and removed.")
    deduped_df = clean_df.drop_duplicates().reset_index(drop=True)


    #Pandera Schema
    #incorporated from notebook

    schema = pa.DataFrameSchema(
    {
        "TYPE": pa.Column(str, pa.Check.isin(crime_types)),
        "YEAR": pa.Column(int, checks=[
            pa.Check.between(2003, 2017),
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'YEAR' column.")],
                          nullable=True),
        "MONTH": pa.Column(int, checks=[
            pa.Check.between(1, 12),
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'MONTH' column.")],
                           nullable=True),
        "DAY": pa.Column(int, checks=[
            pa.Check.between(1, 31), 
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'DAY' column.")],
                            nullable=True),
        "HOUR": pa.Column(float, checks=[
            pa.Check.between(0, 24),
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'DAY' column.")],
                            nullable=True),
        "MINUTE": pa.Column(float, checks=[
            pa.Check.between(0, 60), 
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'MINUTE' column.")],
                            nullable=True),
        "HUNDRED_BLOCK": pa.Column(str, pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'HUNDRED_BLOCK' column."),
                            nullable=True),  
            # no check on specific levels because current dataset has 21204 unique values
        "NEIGHBOURHOOD": pa.Column(str, pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'NEIGHBOURHOOD' column."),
                            nullable=True),
            # no check on specific levels because current dataset has 24 unique values, and new neighbourhoods could be added
        "X": pa.Column(float, checks=[
            pa.Check.between(343000, 615910), 
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'X' column.")],
                            nullable=True),
            # approximate X & Y UTM coordinates chosen from the following map
            # https://coordinates-converter.com/en/decimal/49.120624,-125.156250?karte=OpenStreetMap&zoom=8
        "Y": pa.Column(float, checks=[
            pa.Check.between(5420000, 5530000), 
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'Y' column.")],
                            nullable=True),
        "Latitude": pa.Column(float, checks=[
            pa.Check.between(49, 50),
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'Latitude' column.")],
                            nullable=True),
        "Longitude": pa.Column(float, checks=[
            pa.Check.between(-125, -121), 
            pa.Check(lambda s: s.isna().mean() <= 0.05,
                                    element_wise=False,
                                    error="Too many null values in 'Longitude' column.")],
                            nullable=True)
    },
    checks=[
        pa.Check(
            lambda df: ~(df.isna().all(axis=1)).any(), 
            error="Empty rows found.")
    ],
    drop_invalid_rows=False
)
    
    error_cases = pd.DataFrame()

    # Validate data and handle errors
    try:
        validated_data = schema.validate(deduped_df, lazy=True)
    except pa.errors.SchemaErrors as e:
        error_cases = e.failure_cases

        # Convert error message to a JSON string
        error_message = json.dumps(e.message, indent=2)
        logging.error("\n" + error_message)

    # Filter out invalid rows based on the error cases
    if not error_cases.empty:
        invalid_indices = error_cases["index"].dropna().unique()
        validated_data = (deduped_df.drop(index=invalid_indices).reset_index(drop=True)
        )
    else:
        validated_data = deduped_df

    #saving cleaned and validated data
    validated_data.to_csv(output_csv, index=False)
    print(f"Cleaned and validated data saved to {output_csv}")

if __name__ == '__main__':
    data_validation()