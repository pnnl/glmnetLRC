##' Lasso and elastic-net logistic regression classification with an arbitrary loss function
##'
##' Lasso and elastic-net logistic regression classification (LRC) with an arbitrary, discrete loss function
##' for the classification error
##'
##' \pkg{glmnetLRC} extends the \pkg{glmnet} package for training elastic-net LRCs with a user-specified
##' discrete loss function used to measure the classification error.  Tuning parameters are selected to
##' minimize the expected loss calculated by cross validation.  There are a handful of functions
##' you'll need (along with their associated methods): \code{\link{lossMatrix}}, \code{\link{glmnetLRC}}, 
##' and \code{\link{predict.glmnetLRC}}
##' 
##' @author Landon Sego, Alex Venzin, John Ramey
##' @docType package
##' @name glmnetLRC-package
##' @rdname glmnetLRC-package
NULL
