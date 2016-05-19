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
##' @param object an object of class \code{LRCpred} returned by
##' \code{\link{predict.glmnetLRC}}.
##'
##' @param \dots Arguments passed to \code{\link{print}} methods
##'
##' @return Prints and invisibly returns the following information:
##' \itemize{
##' 
##' \item If \code{truthCol} was provided in the call to
##' \code{\link{predict.glmnetLRC}}, the sensitivity, specificity, false negative rate,
##' false positive rate, and accuracy for the class designated by the second level of
##' the \code{truthLabels} argument
##' in \code{\link{glmnetLRC}} are calculated.  A summary of the predicted probabilities, according to the true class, is
##' also provided.
##'
##' \item If \code{truthCol = NULL} in the call to \code{\link{predict.glmnetLRC}}
##' a tabulation of the number of predictions for each class is shown, along with a summary of the probabilities according
##' to each predicted class.
##' }
##'
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} and \code{\link{glmnetLRC_fit}}
##' for examples.

summary.LRCpred <- function(object, ...) {

  truthCol <- attributes(object)$truthCol

  # If there is no truthCol, summarize the counts of the classified items as well
  # as the distribution of the predicted probabilities
  if (is.null(truthCol)) {

    PredClassSummary <- summary(object$PredictClass)

    PredictedClass <- object$PredictClass
    PredProbSummary <- by(object$Prob, PredictedClass, summary)

    cat("Counts of predicted classes:\n\n")
    print(PredClassSummary, ...)


    cat("\nSummary of predicted probabilities, by predicted class:\n\n")
    print(PredProbSummary, ...)

    invisible(list(PredClassSummary = PredClassSummary,
                   PredProbSummary = PredProbSummary))
    
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

    cat("Logistic regression classification performance:\n\n")
    print(summ_out, ...)

    cat("\nSummary of predicted probabilities, by true class:\n\n")
    print(PredProbSummary, ...)

    invisible(list(ConfusionMatrixMetrics = summ_out,
                   PredProbSummary = PredProbSummary))

  } # else
  
} # end summary.LRCpred
