# Vancouver Crime Predictor

**Authors:** Nicole Link, Tirth Joshi, Zain Nofal

## About

This project analyzes crime patterns in Vancouver using machine learning to predict crime types based on temporal and spatial features. Using data from the Vancouver Police Department spanning 2003-2017, we explore how factors like time of day, location, and neighborhood characteristics can help predict different categories of crime.

Our analysis employs three different machine learning approaches: Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM) to classify crime types. Through exploratory data analysis and model comparison, we aim to understand which features are most predictive and which modeling approach works best for this classification task.

The dataset includes over 530,000 crime incidents across 11 different crime types, ranging from theft and mischief to more serious offenses. By building predictive models, we hope to provide insights that could potentially assist in resource allocation and crime prevention strategies.

## Report

The complete analysis is available in the Jupyter notebook `vancouver_crime_predictor.ipynb` in this repository. The notebook includes:
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
   conda env create -f environment.yml
   conda activate Vancouver_Crime_Predictor
   ```

   This will install all necessary dependencies including Python 3.12, pandas, scikit-learn, and visualization libraries.

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
   - Navigate to `vancouver_crime_predictor.ipynb` in the Jupyter interface
   - Run all cells sequentially (Cell → Run All) or execute cells individually

The notebook will automatically download the latest version of the crime dataset from Kaggle and perform the complete analysis. The entire analysis may take several minutes to complete, particularly the model training sections.

## Dependencies

This project requires the following main dependencies:

- **Python 3.12**
- **Data manipulation:** pandas 2.2, numpy 1.26
- **Machine learning:** scikit-learn 1.5, imbalanced-learn 0.12
- **Visualization:** altair-all 5.4, matplotlib 3.9, seaborn 0.13, folium 0.18
- **Notebook environment:** jupyterlab 4.2, jupyter 1.1, ipykernel, nb_conda_kernels
- **Data access:** kagglehub (via pip)
- **Environment management:** conda-lock 2.5.8
- **HTTP requests:** requests 2.32

For a complete list of dependencies with exact versions, see `environment.yml`. Platform-specific lock files (`conda-lock.yml`) ensure reproducibility across different operating systems.

## License

**Report License:**         
Vancouver Crime Predictor © 2025 by Nicole Link, Tirth Joshi, Zain Nofal is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). This means you can share the report but cannot modify it or use it for commercial purposes.

**Code License:**  
The software code contained within this repository is licensed under the [MIT License](LICENSE). See the LICENSE file for full details.

## References

1. Vancouver Police Department. (2017). *Crime Data* [Dataset]. Kaggle. https://www.kaggle.com/datasets/wosaku/crime-in-vancouver

2. Scikit-learn developers. (2024). *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org/

3. Pandas development team. (2024). *pandas documentation*. https://pandas.pydata.org/docs/

4. Altair developers. (2024). *Altair: Declarative Visualization in Python*. https://altair-viz.github.io/

5. UBC Master of Data Science. (2025-26). *DSCI 571: Supervised Learning I*. University of British Columbia. https://github.com/UBC-MDS

6. AI Coding Assistant: Claude AI agent in VS Code was used for debugging assistance and code optimization. All data analysis, model selection, and interpretation of results were completed independently by the authors.
