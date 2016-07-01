Version 0.1.8, 2016-07-01
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- `coef.glmnetLRC()` receieves a tolerance argument to avoid selecting coefficients that are very close to 0. 


Version 0.1.7, 2016-06-21
-----------------------------------------------------------------------------------

FIXES

- `predict.glmnetLRC()` now correctly handles matrices as predictors, as well as data frames.


Version 0.1.6, 2016-06-17
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Removing use of `Smisc::sortDF()` in `single_glmnetLRC()`


Version 0.1.5, 2016-05-23
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Added `plot.LRCpred()` to plot the predicted probabilites of the LRC, along with associated tests
- In `summary.LRCpred()`: Included summary of predicted probabilities, added `print.summaryLRCpred()`
- Tightened up the argument checking of `predLoss_glmnetLRC()`


Version 0.1.4, 2016-05-11
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Added prediction probabilities to the output of the `predict.glmnetLRC()` method


Version 0.1.3, 2016-05-04
-----------------------------------------------------------------------------------

FIXES

- Corrected the way `predict.glmnetLRC()` handled additional columns specified by `keepCols`

FEATURES / CHANGES

- Added the `missingpreds` methods to easily identify needed predictors that may not be present in new data
- Remove uneeded documentation for generic method, `extract`


Version 0.1.2, 2016-04-13
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Package vignette moved to [online docs](http://pnnl.github.io/docs-glmnetLRC)
- Minor documentation edits


Version 0.1.1, 2016-03-09
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Addition of tests against manual fitting with standardized predictors


Version 0.1.0, 2016-02-15
-----------------------------------------------------------------------------------

FEATURES / CHANGES

- Original package deployment to github
