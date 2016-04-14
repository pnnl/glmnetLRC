##' Calculate performance metrics of logistic regression classifiers
##'
##' Calculate accuracy, sensitivity, specificity, false positive rate, and false
##' negative rate for predictions from logistic regression classifiers
##'
##' @author Landon Sego
##' 
##' @method summary LRCpred
##'
##' @param object an object of class \code{LRCpred} returned by
##' \code{\link{predict.glmnetLRC}}.
##'
##' @param \dots Ignored
##'
##' @return
##' \itemize{
##' \item If \code{truthCol} was provided in the call to
##' \code{\link{predict.glmnetLRC}}, a
##' \code{data.frame} is returned with the sensitivity, specificity, false negative rate,
##' false positive rate, and accuracy for the class designated by the second level of
##' the \code{truthLabels} argument
##' in \code{\link{glmnetLRC}}.
##'
##' \item If \code{truthCol = NULL} in the call to
##' \code{\link{predict.glmnetLRC}}
##' a tabulation of the number of predictions for each class is returned.
##' }
##'
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} and \code{\link{glmnetLRC_fit}}
##' for examples.

summary.LRCpred <- function(object, ...) {

  truthCol <- attributes(object)$truthCol

  # If there are any missing data in the PredictClass or the Truth class,
  # remove them
  cc <- complete.cases(object[, c("PredictClass", truthCol)])

  if (any(!cc)) {

    warning("'object' has ", sum(!cc),
            " missing observation(s) which will be removed\n")

    object <- object[cc,]

  }

  # Calculate the confusion matrix metrics
  summ_out <- cmMetrics(object[, truthCol],
                        object[,"PredictClass"],
                        attributes(object)$classNames[2])

  return(as.data.frame(summ_out))

} # end summary.LRCpred
