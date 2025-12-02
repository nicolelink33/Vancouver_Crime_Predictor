# download_data.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-01

import click
import kagglehub
import os
import zipfile
import pandas as pd

@click.command()
@click.option('--dataset', required=True, help='Kaggle dataset identifier')
@click.option('--output_csv', required=True, help='Path to save the CSV file')
@click.option('--output_zip', required=True, help='Path to save the zipped file')

def download_data(dataset, output_csv, output_zip):
    """Download dataset from Kaggle and save locally as CSV and ZIP."""
    
    path = kagglehub.dataset_download(dataset)
    
    
    df = pd.read_csv(os.path.join(path, "crime.csv"))
    
    df.to_csv(output_csv, index=False)
    
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname=os.path.basename(output_csv))
    
    
    os.remove(output_csv)
    

if __name__ == '__main__':
    download_data()