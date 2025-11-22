# Data Directory

## About the Dataset

This project uses the **Vancouver Crime Dataset** from the Vancouver Police Department, covering crime incidents from 2003 to 2017. The dataset includes over 530,000 records with information about crime types, locations, timestamps, and geographic coordinates.

**Dataset Source:** [Crime in Vancouver - Kaggle](https://www.kaggle.com/datasets/wosaku/crime-in-vancouver)

## Data Download

The data for this project is **automatically downloaded** when you run the analysis notebook (`proj.ipynb`). We use the `kagglehub` library to fetch the latest version of the dataset directly from Kaggle.

### How it works:

1. When you run the first cell of `proj.ipynb`, the code automatically downloads the dataset
2. The data is cached locally on your machine (typically in `~/.cache/kagglehub/`)
3. Subsequent runs will use the cached version unless you explicitly request a fresh download

### First-time setup:

If this is your first time using Kaggle datasets, you may be prompted to authenticate:
- Create a free account at [kaggle.com](https://www.kaggle.com)
- The `kagglehub` library will guide you through the authentication process

### Manual download (optional):

If you prefer to download the data manually:
1. Visit the [dataset page on Kaggle](https://www.kaggle.com/datasets/wosaku/crime-in-vancouver)
2. Download the `crime.csv` file
3. Place it in this `data/` directory
4. Modify the notebook to read from this local path instead of using `kagglehub`

## Data Files

Once downloaded, the dataset consists of:
- **crime.csv** - Main dataset file with all crime records

## Data Dictionary

The dataset includes the following columns:
- `TYPE` - Type of crime (e.g., Theft, Break and Enter, Mischief)
- `YEAR` - Year the crime occurred
- `MONTH` - Month the crime occurred
- `DAY` - Day of the month
- `HOUR` - Hour of the day (24-hour format)
- `MINUTE` - Minute of the hour
- `HUNDRED_BLOCK` - Generalized location (hundred block)
- `NEIGHBOURHOOD` - Vancouver neighborhood
- `X`, `Y` - UTM coordinates
- `Latitude`, `Longitude` - Geographic coordinates

## Notes

- This directory may remain empty in the repository since data is downloaded automatically
- The dataset is approximately 60 MB in size
- Data is provided by the Vancouver Police Department and hosted on Kaggle
