##' A LRCbestsubsets model fit object
##'
##' @description
##' This object is returned by the call
##' \code{LRCbestsubsets_fit <- LRCbestsubsets(cheat, predictors, lM,
##'                                            cvReps = 100, cvFolds = 5,
##'                                            cores = max(1, detectCores() - 1))}
##' in the example of \code{\link{LRCbestsubsets}}. It is preserved here in the package
##' because it is time consuming to generate.
##'
##' @docType data
##' @format A LRCbestsubsets object returned by \code{\link{LRCbestsubsets}}
##' @name LRCbestsubsets_fit
##' @examples
##' # Load fitted LRCbestsubsets model
##' data(LRCbestsubsets_fit)
##'
##' # Show the optimal parameter values
##' print(LRCbestsubsets_fit)
##'
##' # Show the plot of all the optimal parameter values for each cross validation replicate
##' plot(LRCbestsubsets_fit)
##'
##' # Load the training set (ideally a testing set would be best)
##' data(Mojave)
##'
##' # Use the trained model to make predictions about
##' # new observations for the response variable.
##' new <- predict(LRCbestsubsets_fit, Mojave, truthCol = "cheatGrass", keepCols = 1:2)
##' head(new)
##'
##' # Now summarize the performance of the model
##' summary(new)
##'
##' # If predictions are made without the an indication of the ground truth,
##' # the summary is simpler:
##' summary(predict(LRCbestsubsets_fit, Mojave))
NULL
