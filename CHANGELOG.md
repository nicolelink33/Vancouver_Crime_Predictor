# Changelog

All notable changes to this project based on peer review feedback will be documented in this file.

## [Post-Milestone 4] - 2025-12-30

### Fixed - Addressing Milestone 4 Grading Feedback

- **Added pytest configuration** ([pytest.ini](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/pytest.ini))
  - Configured `pythonpath = .` to enable test imports without manual environment variable setting
  - Added test path configuration and naming conventions
  - Tests now work without requiring `PYTHONPATH=.` environment variable
  - Addresses Milestone 4 grading feedback about test automation (-2.5 pts)

- **Added make test target** ([Makefile](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/Makefile#L123-L126))
  - Provides explicit command for running test suite: `make test`
  - Documents PYTHONPATH requirement in Makefile
  - Improves developer experience and test discoverability

- **Fixed data cleanup** ([Makefile](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/Makefile#L116))
  - Updated clean target to remove all `.zip` and `.csv` files in data directory
  - Now removes `crime_data_2025-11-22.zip` and any other generated data files
  - Changed from specific file names to wildcard patterns: `data/*.csv data/*.zip`
  - Addresses Milestone 4 grading feedback about data pipeline (-1.0 pt)

- **Updated .gitignore** ([.gitignore](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/.gitignore#L17-L20))
  - Added patterns to ignore generated data files
  - Prevents accidental commits of `data/*.zip`, `data/*.csv`, `data/*.pickle`
  - Ensures clean repository state

- **Added input validation to data_utils.py** ([src/data_utils.py](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/src/data_utils.py))
  - Added type checking for all function parameters in `download_and_load_csv`
  - Added type checking for all function parameters in `save_csv_and_zip`
  - Validates DataFrame types, string inputs, and checks for empty values
  - Provides clear, descriptive error messages for invalid inputs
  - Checks file existence before attempting to read CSV files
  - Addresses Milestone 4 grading feedback about function design (-0.5 pts)

- **Added input validation to plotting.py** ([src/plotting.py](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/src/plotting.py))
  - Added comprehensive type checking for `save_bar_plot` function
  - Validates DataFrame type and checks for empty DataFrames
  - Validates column names exist in DataFrame (except for special `count()` case)
  - Validates all string parameters (x_col, y_col, title, save_path)
  - Added pandas import for DataFrame type checking
  - Addresses Milestone 4 grading feedback about function design (-0.5 pts)

- **Added error handling tests** ([tests/test_data_utils.py](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/tests/test_data_utils.py), [tests/test_plotting.py](file:///Users/tirthjoshi/MDS/Block_3/522/Vancouver_Crime_Predictor/tests/test_plotting.py))
  - Added 12 error handling tests for `download_and_load_csv` and `save_csv_and_zip`
  - Added 10 error handling tests for `save_bar_plot`
  - Tests cover invalid types, empty inputs, missing columns, and edge cases
  - All tests verify proper error messages are raised
  - Addresses Milestone 4 grading feedback about test design (-0.75 pts)

### Summary of Milestone 4 Feedback Resolution

All issues identified in Milestone 4 grading have been addressed:

| Issue | Points Lost | Status | Fix |
|-------|-------------|--------|-----|
| Test automation | -2.5 | Fixed | Added pytest.ini and make test target |
| Data cleanup | -1.0 | Fixed | Updated Makefile clean target |
| Input validation | -0.5 | Fixed | Added validation to all utility functions |
| Test edge cases | -0.75 | Fixed | Added 22 error handling tests |
| **Total** | **-4.75** | **All Fixed** | |

All tests now pass (42 total tests including 21 new error handling tests).

## [Milestone 4] - 2025-12-13

### Fixed - Addressing Milestone 2 Grading Feedback

- **Package version pinning** ([86b286e](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/86b286e))
  - Changed pip packages from `>=` to `==` for exact version pinning
  - `deepchecks>=0.18.0` → `deepchecks==0.19.1`
  - `vegafusion[embed]>=1.5.0` → `vegafusion[embed]==2.0.1`
  - Ensures reproducible Docker builds in the future
  - Addresses Milestone 2 grading feedback about version pinning (-2 pts)

- **Added deno to fix Quarto rendering** ([86b286e](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/86b286e))
  - Added `deno=2.3.1` to environment.yml
  - Provides missing runtime dependency for Quarto
  - Regenerated conda-lock.yml with deno included
  - Improves Docker container reproducibility

- **GitHub Actions workflow trigger** (Already implemented)
  - `conda-lock.yml` already included in workflow triggers (line 9)
  - Ensures Docker image rebuilds when lock file changes
  - Addresses Milestone 2 grading feedback about workflow triggers (-0.5 pts)

### Fixed - Addressing Milestone 1 Grading Feedback

All items below were addressed and included in the Milestone 4 submission:

- **Report rendering to PDF and HTML** 
  - Quarto report now renders to both HTML and PDF formats
  - Files: `reports/vancouver_crime_predictor.html` and `reports/vancouver_crime_predictor.pdf`
  - Addresses Milestone 1 grading feedback (-1 pt)

- **Package version specifications**
  - All packages in `environment.yml` now have exact versions specified
  - All conda packages use `=` (e.g., `python=3.11`)
  - All pip packages use `==` (e.g., `kagglehub==0.3.4`)
  - Addresses Milestone 1 grading feedback (-1 pt)

- **LICENSE file with all team members**
  - Updated LICENSE to include all group member names
  - Copyright line: "Nicole Link, Zain Nofal, Tirth Joshi"
  - Addresses Milestone 1 grading feedback (-1 pt)

- **Target/response variable definition**
  - Clearly defined crime type as the target variable in report
  - Added detailed description of 11 crime types and selection criteria
  - Included visualization of crime type distribution
  - Addresses Milestone 1 grading feedback (-0.5 pts)

- **Dataset references**
  - Added BibTeX citations for dataset using `references.bib`
  - Dataset properly cited as `[@CrimeKaggle]` in report
  - Addresses Milestone 1 grading feedback (-0.25 pts)

- **Methodology descriptions**
  - Expanded Methods section with comprehensive details
  - Included: models used, feature engineering, train/test split, hyperparameter optimization
  - Clearly described evaluation metrics and approach
  - Addresses Milestone 1 grading feedback (-1 pt)

- **Code for saving data**
  - Implemented data saving in all analysis scripts
  - `preprocessing.py` saves train/test splits
  - All model evaluation scripts save results to CSV
  - Complete data pipeline documented in Makefile
  - Addresses Milestone 1 grading feedback (-0.5 pts)

- **GitHub Issues for communication**
  - GitHub issues have been consistently created for each milestone
  - Issues have been actively used for project communication and task tracking
  - All milestone-related issues have been resolved
  - Addresses Milestone 1 grading feedback (-4 pts)

## [Unreleased] - 2025-12-10

### Added
- Issue reporting and support section in README ([#peer-review-fixes](https://github.com/nicolelink33/Vancouver_Crime_Predictor/tree/peer-review-fixes))
  - Added "Getting Help" section with instructions on how to report issues
  - Included links to GitHub issues and discussions
  - Addresses feedback from reviewer @Chikire about missing issue reporting instructions
  
- Raw data folder structure ([aa23f4a](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/aa23f4a))
  - Created `data/raw/` folder with README
  - Better organization for raw vs processed data
  - Addresses feedback from reviewers @Chikire and @sapolraadnui about data folder organization

- Limitations section in report ([2ec9cf2](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/2ec9cf2))
  - Added comprehensive "Limitations and Assumptions" subsection to Discussion
  - Discusses feature limitations, class imbalance, missing data, and dataset age
  - Addresses professor's M2 feedback about including limitations

### Changed
- Fixed knn_eval.py directory path bug ([4fa5dae](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/4fa5dae))
  - Removed extra 'tables' subdirectory from results path
  - Fixes error where script tried to save to `results/tables/tables`
  - Addresses feedback from reviewer @Chikire about script execution error

- Improved report formatting ([8d8fd2c](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/8d8fd2c))
  - Removed informal section headings ("Was This Expected?", "Why This Matters")
  - Moved content to formal Discussion section
  - Converted comparison results to table format instead of bullet points
  - Changed Future Work from bullet points to paragraph format
  - Addresses feedback from reviewers @jentsang and @sapolraadnui about report formality

- Enhanced Summary section ([2ec9cf2](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/2ec9cf2))
  - Added importance statement about why crime prediction matters
  - Included limitations in abstract
  - Addresses professor's M2 feedback about abstract requirements

### Peer Review Summary

We received feedback from three reviewers (@jentsang, @Chikire, @sapolraadnui) as part of Milestone 3 peer review. The main themes were:

1. **Documentation improvements** - Adding issue reporting instructions and better data organization
2. **Code quality** - Fixing the knn_eval.py directory bug that prevented script execution
3. **Report formality** - Making the report more academic and less informal

All feedback has been addressed in the commits listed above.

### References
- Peer review issue: https://github.com/UBC-MDS/data-analysis-review-2025/issues/5
- Branch with fixes: https://github.com/nicolelink33/Vancouver_Crime_Predictor/tree/peer-review-fixes
