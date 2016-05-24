##' Summarize predictions of logistic regression classifier
##'
##' Summarize the predicted probabilities of the classifier, and, if possible,
##' calculate accuracy, sensitivity, specificity, false positive rate, and false
##' negative rate.
##' 
##' @author Landon Sego
##' 
##' @method summary LRCpred
##'
##' @param object An object of class \code{LRCpred} returned by
##' \code{\link{predict.glmnetLRC}}.
##'
##' @param \dots Arguments passed to \code{\link{print}} methods. Ignored by the \code{summary} method.
##'
##' @return Returns a \code{summaryLRCpred} object.  If \code{truthCol} was provided in the call to
##' \code{\link{predict.glmnetLRC}}, the result is a list with the following elements:
##'
##' \describe{
##' \item{ConfusionMatrixMetrics}{A matrix with the sensitivity, specificity, false negative rate,
##' false positive rate, and accuracy for the class designated by the second level of
##' the \code{truthLabels} argument provided to \code{\link{glmnetLRC}}}
##' \item{PredProbSummary}{A numeric summary of the predicted probabilities, according to the true class}
##' }
##'
##' If \code{truthCol} was not provided in the call to \code{\link{predict.glmnetLRC}}, the result is a list with
##' the following elements:
##' \describe{
##' \item{PredClassSummary}{A tabulation of the number of predictions in each class}
##' \item{PredProbSummary}{A numeric summary of the predicted probabilities, according to the predicted class}
##' }
##'
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} for examples.

summary.LRCpred <- function(object, ...) {

  truthCol <- attributes(object)$truthCol

  # If there is no truthCol, summarize the counts of the classified items as well
  # as the distribution of the predicted probabilities
  if (is.null(truthCol)) {

    PredClassSummary <- summary(object$PredictClass)

    PredictedClass <- object$PredictClass
    PredProbSummary <- by(object$Prob, PredictedClass, summary)

    out <- list(PredClassSummary = PredClassSummary,
                PredProbSummary = PredProbSummary)

  }

  else {

    TrueClass <- object[,truthCol]
    PredProbSummary <- by(object$Prob, TrueClass, summary)
      
    # If there are any missing data in the PredictClass or the Truth class,
    # remove them
    cc <- complete.cases(object[, c("PredictClass", truthCol)])
  
    if (any(!cc)) {
  
      warning("'object' has ", sum(!cc),
              " missing observation(s) which will be removed in the summary\n")
  
      object <- object[cc,]
  
    }
  
    # Calculate the confusion matrix metrics
    summ_out <- cmMetrics(object[, truthCol],
                          object[,"PredictClass"],
                          attributes(object)$classNames[2])

    out <- list(ConfusionMatrixMetrics = summ_out,
                PredProbSummary = PredProbSummary)

  } # else

  class(out) <- c("summaryLRCpred", class(out))

  return(out)
  
} # end summary.LRCpred


##' @method print summaryLRCpred
##'
##' @describeIn summary.LRCpred Prints a \code{summaryLRCpred} object in a readable format.
##'
##' @param x An object of class \code{summaryLRCpred}.
##'
##' @export

print.summaryLRCpred <- function(x, ...) {

  if ("ConfusionMatrixMetrics" %in% names(x)) {

    cat("Logistic regression classification performance:\n\n")
    print(x$ConfusionMatrixMetrics, ...)

    cat("\nSummary of predicted probabilities, by true class:\n\n")
    print(x$PredProbSummary, ...)
      
  }
  else {
      
    cat("Counts of predicted classes:\n\n")
    print(x$PredClassSummary, ...)

    cat("\nSummary of predicted probabilities, by predicted class:\n\n")
    print(x$PredProbSummary, ...)

  }
    
}
