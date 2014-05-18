##' A glmnetLRC model fit object
##'
##' @description
##' This object is returned by the call
##' \code{glmnetLRC_fit <- glmnetLRC(response, predictors, lM, cores = max(1, detectCores() - 1))}
##' in the example of \code{\link{glmnetLRC}}. It is preserved here in the package
##' because it is time consuming to generate.
##'
##' @docType data
##' @format A glmnetLRC object returned by \code{\link{glmnetLRC}}
##' @name glmnetLRC_fit
##' @examples
##' # Load fitted glmnetLRC model
##' data(glmnetLRC_fit)
##' 
##' # Show the optimal parameter values
##' print(glmnetLRC_fit)
##'
##' # Show the plot of all the optimal parameter values for each cross validation replicate
##' plot(glmnetLRC_fit)
##'
##' # Load the new observations
##' data(testdata)
##'
##' # Use the trained model to make predictions about
##' # new observations for the response variable.
##' new <- predict(glmnetLRC_fit, testdata, truthCol = "Curated_Quality", keepCols = 1:2)
##' head(new)
##'
##' # Now summarize the performance of the model
##' summary(new)
##'
##' # If predictions are made without the an indication of the ground truth,
##' # the summary is simpler:
##' summary(predict(glmnetLRC_fit, testdata))
NULL
