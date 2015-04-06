##' Calculate performance metrics of logistic regression classifiers
##'
##' Calculate accuracy, sensitivity, specificity, false positive rate, and false
##' negative rate for predictions from logistic regression classifiers
##'
##' @author Landon Sego
##' 
##' @method summary LRCpred
##'
##' @param LRCpredObject an object of class \code{LRC_pred} returned by
##' \code{\link{predict.LRCbestsubsets}} or \code{\link{predict.LRCglmnet}}.
##'
##' @return
##' \itemize{
##' \item If \code{truthCol} was provided in the call to
##' \code{\link{predict.LRCglmnet}} or \code{\link{predict.LRCbestsubsets}}, a
##' \code{data.frame} is returned with the sensitivity, specificity, false negative rate,
##' false positive rate, and accuracy for the class designated by the second level of
##' the \code{truthLabels} argument in \code{\link{LRCglmnet}} or
##' \code{\link{LRCbestsubsets}}.
##'
##' \item If \code{truthCol = NULL} in the call to \code{\link{predict.LRCglmnet}},
##' or \code{\link{predict.LRCbestsubsets}}, a tabulation
##' of the number of predictions for each class is returned.
##' }
##'
##' @export

summary.LRCpred <- function(LRCpredObject) {

  truthCol <- attributes(LRCpredObject)$truthCol

  # If there are any missing data in the PredictClass or the Truth class,
  # remove them
  cc <- complete.cases(LRCpredObject[, c("PredictClass", truthCol)])

  if (any(!cc)) {

    warning("'LRCpredObject' has ", sum(!cc),
            " missing observation(s) which will be removed\n")

    LRCpredObject <- LRCpredObject[cc,]

  }

  # Calculate the confusion matrix
  confusion_matrix <- confusion(LRCpredObject[, truthCol],
                                LRCpredObject[,"PredictClass"])

  # calculate sensitivity
  sens <- sensitivity(confusion_matrix)

  # calculate specificity
  spec <- specificity(confusion_matrix)

  # calculate accuracy
  acc <- accuracy(confusion_matrix)

  # false negative rate = 1 - TPR = 1 - sensitivity
  FNR <- function(confusionSummary, aggregate = c('micro', 'macro')){
    sens <- sensitivity(confusionSummary)
    return(1 - sens$byClass)
  }

  # false positive rate = 1 - TNR = 1 - specificity
  FPR <- function(confusionSummary, aggregate = c('micro', 'macro')){
    spec <- specificity(confusionSummary)
    return(1 - spec$byClass)
  }

  # calculate fnr
  fnr <- FNR(confusion_matrix)

  # calculate fpr
  fpr <- FPR(confusion_matrix)

  # put it into a data.frame format
  summ_out <- rbind(sens$byClass, spec$byClass, fnr, fpr, acc$byClass)

  # Only show the second class--the target of the prediction
  summ_out <- matrix(summ_out[,attributes(LRCpredObject)$classNames[2]], ncol = 1)

  colnames(summ_out) <- attributes(LRCpredObject)$classNames[2]

  rownames(summ_out) <- c("sensitivity", "specificity",
                          "false negative rate", "false positive rate",
                          "accuracy")

  return(as.data.frame(summ_out))

} # end summary.LRCpred