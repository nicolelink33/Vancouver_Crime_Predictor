# Changelog

All notable changes to this project based on peer review feedback will be documented in this file.

## [Unreleased] - 2025-12-10

### Added
- Issue reporting and support section in README ([#peer-review-fixes](https://github.com/nicolelink33/Vancouver_Crime_Predictor/tree/peer-review-fixes))
  - Added "Getting Help" section with instructions on how to report issues
  - Included links to GitHub issues and discussions
  - Addresses feedback from reviewer @Chikire about missing issue reporting instructions
  
- Raw data folder structure ([2cc2ecb](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/2cc2ecb))
  - Created `data/raw/` folder with README
  - Better organization for raw vs processed data
  - Addresses feedback from reviewers @Chikire and @sapolraadnui about data folder organization

- Limitations section in report ([2cc2ecb](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/2cc2ecb))
  - Added comprehensive "Limitations and Assumptions" subsection to Discussion
  - Discusses feature limitations, class imbalance, missing data, and dataset age
  - Addresses professor's M2 feedback about including limitations

### Changed
- Fixed knn_eval.py directory path bug ([9f88e8d](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/9f88e8d))
  - Removed extra 'tables' subdirectory from results path
  - Fixes error where script tried to save to `results/tables/tables`
  - Addresses feedback from reviewer @Chikire about script execution error

- Improved report formatting ([749d2d9](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/749d2d9))
  - Removed informal section headings ("Was This Expected?", "Why This Matters")
  - Moved content to formal Discussion section
  - Converted comparison results to table format instead of bullet points
  - Changed Future Work from bullet points to paragraph format
  - Addresses feedback from reviewers @jentsang and @sapolraadnui about report formality

- Enhanced Summary section ([2cc2ecb](https://github.com/nicolelink33/Vancouver_Crime_Predictor/commit/2cc2ecb))
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
