.PHONY: all clean

all: reports/vancouver_crime_predictor.html reports/vancouver_crime_predictor.pdf

# Step 1: Download data
data/ :
	mkdir -p data

data/crimedata.csv data/crimedata.zip : scripts/download_data.py | data/
	PYTHONPATH=. python scripts/download_data.py \
		--dataset=wosaku/crime-in-vancouver \
		--output-csv=data/crimedata.csv \
		--output-zip=data/crimedata.zip

# Step 2: Clean data
data/crimedata_clean.csv : data/crimedata.csv scripts/data_validation.py
	python scripts/data_validation.py \
		--input-csv=data/crimedata.csv \
		--output-csv=data/crimedata_clean.csv

# Step 3: Preprocess
data/processed/X_train.csv data/processed/y_train.csv data/processed/X_test.csv data/processed/y_test.csv data/preprocessor.pickle : \
data/crimedata_clean.csv scripts/preprocessing.py
	python scripts/preprocessing.py \
		--raw-data=data/crimedata_clean.csv \
		--data-to=data/processed \
		--preprocessor-to=data/preprocessor.pickle \
		--seed=522

# Step 4: EDA Figures
results/figures/eda_done.flag results/figures/crime_type_distribution.png : data/crimedata_clean.csv scripts/eda.py
	PYTHONPATH=. python scripts/eda.py \
		--processed-training-data=data/crimedata_clean.csv \
		--target-csv=data/crimedata_clean.csv \
		--plot-to=results/figures
	touch results/figures/eda_done.flag

# Step 5: KNN
results/models/knn_model.pickle results/models/knn_model_baseline.pickle results/figures/knn_k_optimization.png : \
data/processed/X_train.csv data/processed/y_train.csv scripts/knn_training.py
	python scripts/knn_training.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--model-out=results/models/knn_model.pickle \
		--plot-out=results/figures/knn_k_optimization.png \
		--seed=522

results/figures/knn_confusion_matrix.png results/tables/knn_class_report.txt results/tables/knn_score.csv results/tables/knn_baseline_score.csv : \
data/processed/X_test.csv data/processed/y_test.csv results/models/knn_model.pickle results/models/knn_model_baseline.pickle scripts/knn_eval.py
	PYTHONPATH=. python scripts/knn_eval.py \
        --x-test-path=data/processed/X_test.csv \
        --y-test-path=data/processed/y_test.csv \
        --model-path=results/models/knn_model.pickle \
        --plot-out=results/figures/knn_confusion_matrix.png \
        --report-out=results/tables/knn_class_report.txt

# Step 6: SVM training
results/models/svm_final_best_fit.pickle results/models/svm_baseline_fit.pickle \
results/models/svm_initial_grid_fit.pickle results/models/svm_final_random_fit.pickle \
results/figures/svm_initial_grid_fit.png results/figures/svm_final_random_fit.png : \
data/processed/X_train.csv data/processed/y_train.csv data/preprocessor.pickle scripts/svm_training.py
	python scripts/svm_training.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--preprocessor=data/preprocessor.pickle \
		--pipeline-to=results/models \
		--plot-to=results/figures \
		--seed=522

# Step 7: SVM evaluation
results/tables/svm_baseline_score.csv results/tables/svm_score.csv \
results/tables/svm_class_report.csv results/figures/svm_confusion_matrix.png : \
data/processed/X_test.csv data/processed/y_test.csv scripts/svm_eval.py \
results/models/svm_final_best_fit.pickle results/models/svm_baseline_fit.pickle
	PYTHONPATH=. python scripts/svm_eval.py \
        --x-test-path=data/processed/X_test.csv \
        --y-test-path=data/processed/y_test.csv \
        --pipeline-from=results/models \
        --results-to=results \
        --plot-to=results/figures

# Step 8: Logistic Regression
results/models/log_reg_model.pickle results/models/logreg_baseline_fit.pickle results/tables/logreg_params.csv : \
data/processed/X_train.csv data/processed/y_train.csv scripts/log_reg_fit.py
	python scripts/log_reg_fit.py \
		--x-train-path=data/processed/X_train.csv \
		--y-train-path=data/processed/y_train.csv \
		--model-out=results/models \
		--params-out=results/tables \
		--seed=522

results/tables/logreg_score.csv results/tables/logreg_baseline_score.csv \
results/figures/log_reg_confusion_matrix.png : \
data/processed/X_test.csv data/processed/y_test.csv scripts/log_reg_eval.py \
results/models/log_reg_model.pickle results/models/logreg_baseline_fit.pickle
	PYTHONPATH=. python scripts/log_reg_eval.py \
        --x-test-path=data/processed/X_test.csv \
        --y-test-path=data/processed/y_test.csv \
        --model-path=results/models \
        --plot-out=results/figures/log_reg_confusion_matrix.png \
        --report-out=results/tables

# Step 9: Final Report
reports/vancouver_crime_predictor.html reports/vancouver_crime_predictor.pdf : \
	reports/vancouver_crime_predictor.qmd reports/references.bib \
	results/tables/svm_baseline_score.csv results/tables/svm_score.csv \
	results/tables/logreg_score.csv results/tables/logreg_baseline_score.csv \
    results/tables/knn_score.csv results/tables/knn_baseline_score.csv \
	results/figures/knn_k_optimization.png results/figures/knn_confusion_matrix.png \
	results/figures/svm_confusion_matrix.png results/figures/log_reg_confusion_matrix.png \
    results/figures/crime_type_distribution.png
	quarto render reports/vancouver_crime_predictor.qmd

# Clean
clean :
	rm -rf data/crimedata.csv data/crimedata_clean.csv data/crime_processed.csv data/crimedata.zip
	rm -rf data/processed/*
	rm -rf data/preprocessor.pickle
	rm -rf results/models/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf reports/vancouver_crime_predictor.pdf reports/vancouver_crime_predictor.html
