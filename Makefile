.PHONY: all clean

all: reports/vancouver_crime_predictor.html reports/vancouver_crime_predictor.pdf

# can we run cd work like this? or do we need to add work/ to all our file pathways? 
# Or in the instructions just tell them to cd work before running make all or make clean?
cd work

# Step 1: Download data from Kaggle
data/crimedata.csv data/crimedata.zip : wosaku/crime-in-vancouver scripts/download_data.py
	python scripts/download_data.py \
		--dataset=wosaku/crime-in-vancouver \
		--output-csv=data/crimedata.csv \
		--output-zip=data/crimedata.zip

# Step 2: Validate and clean the data
data/crimedata_clean.csv : data/crimedata.csv scripts/data_validation.py
	python scripts/data_validation.py \
		--input-csv=data/crimedata.csv \
		--output-csv=data/crimedata_clean.csv

# Step 3: Preprocess data and split into train/test sets
data data/preprocessor.pickle : data/crimedata_clean.csv seed scripts/preprocessing.py
	python scripts/preprocessing.py \
		--raw-data=data/crimedata_clean.csv \
		--data-to=data \
		--preprocessor-to=data/preprocessor.pickle \
		--seed=522

# Step 4: Generate exploratory data analysis visualizations
# Note: EDA uses clean data before preprocessing to access original columns

results/figures : data/crimedata_clean.csv scripts/eda.py
	python scripts/eda.py \
		--processed-training-data=data/crimedata_clean.csv \
		--target-csv=data/crimedata_clean.csv \
		--plot-to=results/figures

# Step 5: Train KNN model
results/models/knn_model.pickle results/figures/knn_k_optimization.png : data/processed/X_train.csv data/processed/y_train.csv seed scripts/knn_training.py
	python scripts/knn_training.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--model-out=results/models/knn_model.pickle \
		--plot-out=results/figures/knn_k_optimization.png \
		--seed=522

# Step 6: Evaluate KNN model
results/figures/knn_confusion_matrix.png results/tables/knn_class_report.txt : data/processed/X_test.csv data/processed/y_test.csv results/models/knn_model.pickle scripts/knn_eval.py
	python scripts/knn_eval.py \
		--x-test-path=data/processed/X_test.csv \
		--y-test-path=data/processed/y_test.csv \
		--model-path=results/models/knn_model.pickle \
		--plot-out=results/figures/knn_confusion_matrix.png \
		--report-out=results/tables/knn_class_report.txt

# Step 7: Train SVM model
results/models results/figures : data/processed/X_train.csv data/processed/y_train.csv seed scripts/svm_training.py
	python scripts/svm_training.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--preprocessor=data/preprocessor.pickle \
		--pipeline-to=results/models \
		--plot-to=results/figures \
		--seed=522

# Step 8: Evaluate SVM model
results/models results results/figures : data/processed/X_test.csv data/processed/y_test.csv scripts/svm_eval.py
	python scripts/svm_eval.py \
		--x-test-path=data/processed/X_test.csv \
		--y-test-path=data/processed/y_test.csv \
		--pipeline-from=results/models \
		--results-to=results \
		--plot-to=results/figures

# Step 9: Train Logistic Regression model
results/models results/tables : data/processed/X_train.csv data/processed/y_train.csv seed scripts/log_reg_fit.py
	python scripts/log_reg_fit.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--model-out=results/models \
		--params-out=results/tables \
		--seed=522

# Step 10: Evaluate Logistic Regression model
results/models results/figures/log_reg_confusion_matrix.png results/tables : data/processed/X_test.csv data/processed/y_test.csv scripts/log_reg_eval.py
	python scripts/log_reg_eval.py \
		--x-test-path=data/processed/X_test.csv \
		--y-test-path=data/processed/y_test.csv \
		--model-path=results/models \
		--plot-out=results/figures/log_reg_confusion_matrix.png \
		--report-out=results/tables

# Step 11: Render the final report
reports/vancouver_crime_predictor.html reports/vancouver_crime_predictor.pdf : reports/vancouver_crime_predictor.qmd \
reports/references.bib results/tables/svm_baseline_score.csv results/tables/svm_score.csv \
results/tables/knn_score.csv results/tables/knn_baseline_score.csv results/tables/logreg_score.csv \
results/tables/logreg_baseline_score.csv results/figures/crime_type_distribution.png \
results/figures/knn_k_optimization.png results/figures/knn_confusion_matrix.png \
results/figures/svm_confusion_matrix.png results/figures/log_reg_confusion_matrix.png
	quarto render reports/vancouver_crime_predictor.qmd

# Clean up analysis
clean :
	rm -rf data/crimedata.csv data/crimedata_clean.csv data/crime_processed.csv data/crimedata.zip
	rm -rf data/processed/*
	rm -rf data/preprocessor.pickle
	rm -rf results/models/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf reports/vancouver_crime_predictor.pdf reports/vancouver_crime_predictor.html