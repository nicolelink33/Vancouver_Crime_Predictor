# Vancouver Crime Predictor

**Authors:** Nicole Link, Tirth Joshi, Zain Nofal

## About

This project analyzes crime patterns in Vancouver using machine learning to predict crime types based on temporal and spatial features. Using data from the Vancouver Police Department spanning 2003-2017, we explore how factors like time of day, location, and neighborhood characteristics can help predict different categories of crime.

Our analysis employs three different machine learning approaches: Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM) to classify crime types. Through exploratory data analysis and model comparison, we aim to understand which features are most predictive and which modeling approach works best for this classification task.

The dataset includes over 530,000 crime incidents across 11 different crime types, ranging from theft and mischief to more serious offenses. By building predictive models, we hope to provide insights that could potentially assist in resource allocation and crime prevention strategies.

## Report

The complete analysis is available in the Jupyter notebook `reports/vancouver_crime_predictor.ipynb` in this repository. The notebook includes:
- Data loading and preprocessing
- Exploratory data analysis with visualizations
- Feature engineering and model training
- Model evaluation and comparison
- Discussion of findings

To view the notebook, you can either:
- Open it directly on GitHub (limited interactivity)
- Run it locally following the instructions in the [Usage](#usage) section below

## Usage

### Setting up the environment

1. **Clone this repository:**
```bash
git clone https://github.com/nicolelink33/Vancouver_Crime_Predictor.git
cd Vancouver_Crime_Predictor
```

2. **Create and activate the conda environment:**
```bash
conda-lock install --name vancouver_crime_predictor conda-lock.yml
conda activate vancouver_crime_predictor
```

   This will install all necessary dependencies including Python 3.11, pandas, scikit-learn, and visualization libraries.
The notebook will automatically download the latest version of the crime dataset from Kaggle and perform the complete analysis. The entire analysis may take several minutes to complete, particularly the model training sections.

3. **Set up Kaggle authentication (required if want to download data):**
   
   The dataset is automatically downloaded from Kaggle using `kagglehub`. On first run, you may be prompted to authenticate with Kaggle. If you don't have a Kaggle account:
   - Create one at [kaggle.com](https://www.kaggle.com)
   - The notebook will handle the rest automatically

### Running the analysis

1. **Launch Jupyter Lab:**
```bash
jupyter lab
```

2. **Open and run the notebook:**
   - Navigate to `reports/vancouver_crime_predictor.ipynb` in the Jupyter interface
   - Under Switch/Select Kernel choose "Python [conda env:vancouver_crime_predictor]"
   - Run all cells sequentially (Cell → Run All) or execute cells individually


## Running with Docker

For a fully reproducible environment, you can run this analysis using Docker.



### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Using Docker Compose (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/nicolelink33/Vancouver_Crime_Predictor.git
cd Vancouver_Crime_Predictor
```

2. Launch the container:
```bash
docker-compose up
```

3. Open Jupyter Lab in your browser at: http://localhost:10000/lab
4. To run the analysis, open a terminal and run the following commands: 
```



python scripts/svm_training.py \
    --X_train_path=data/processed/X_train.csv \
    --y_train_path=data/processed/y_train.csv \
    --preprocessor=results/models/preprocessor.pickle \
    --pipeline-to=results/models \
    --plot-to=results/figures \
    --seed=522

python scripts/svm_small_data_test.py \
    --X_train_path=data/processed/X_train.csv \
    --y_train_path=data/processed/y_train.csv \
    --pipeline-to=results/models \
    --plot-to=results/figures \
    --seed=522

python scripts/svm_eval.py \
    --X_test_path=data/processed/X_test.csv \
    --y_test_path=data/processed/y_test.csv \
    --pipeline-from=results/models \
    --results-to=results/tables \
    --plot-to=results/figures \




quarto render

```
5. Navigate to the `work` folder, then open `reports/vancouver_crime_predictor.ipynb` and run the analysis

6. To stop the container, press `Ctrl+C` in the terminal

### Pulling the Docker Image Directly

You can also pull the pre-built image from DockerHub:
```bash
docker pull tirthjoship/vancouver-crime-predictor:latest
docker run -p 10000:8888 -v $(pwd):/home/jovyan/work tirthjoship/vancouver-crime-predictor:latest
```
### Updating the Docker Image

When dependencies in `environment.yml` change:
- **Python 3.11**

1. The GitHub Actions workflow automatically rebuilds and pushes the image to DockerHub
2. Pull the latest image: `docker pull tirthjoship/vancouver-crime-predictor:latest`
3. Regenerate the lock file if needed: `conda-lock -f environment.yml --lockfile conda-lock.yml`

## Dependencies

This project requires the following main dependencies:

- **Data manipulation:** pandas 2.2, numpy 1.26
- **Machine learning:** scikit-learn 1.5, imbalanced-learn 0.12
- **Visualization:** altair 5.4, matplotlib 3.9, seaborn 0.13, folium 0.18
- **Notebook environment:** jupyterlab 4.2, jupyter 1.1, ipykernel, nb_conda_kernels
- **Data access:** kagglehub (via pip)
- **Data validation:** pandera 0.26.1, deepchecks >=0.18.0
- **Environment management:** conda-lock 2.5.8
- **HTTP requests:** requests 2.32

For a complete list of dependencies with exact versions, see `environment.yml`. Platform-specific lock files (`conda-lock.yml`) ensure reproducibility across different operating systems.
2. Scikit-learn developers. (2024). *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org/

## License

**Report License:**         
Vancouver Crime Predictor © 2025 by Nicole Link, Tirth Joshi, Zain Nofal is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). This means you can share the report but cannot modify it or use it for commercial purposes.

**Code License:**  
The software code contained within this repository is licensed under the [MIT License](LICENSE). See the LICENSE file for full details.

## References

1. Vancouver Police Department. (2017). *Crime Data* [Dataset]. Kaggle. https://www.kaggle.com/datasets/wosaku/crime-in-vancouver


3. Pandas development team. (2024). *pandas documentation*. https://pandas.pydata.org/docs/

4. Altair developers. (2024). *Altair: Declarative Visualization in Python*. https://altair-viz.github.io/

5. UBC Master of Data Science. (2025-26). *DSCI 571: Supervised Learning I*. University of British Columbia. https://github.com/UBC-MDS

6. AI Coding Assistant: Claude AI agent in VS Code was used for debugging assistance and code optimization. All data analysis, model selection, and interpretation of results were completed independently by the authors.
