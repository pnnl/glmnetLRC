# THIS IS SQM package stuff--eventually, I'll want to call these functions with the SQM
# package

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

# Calculate confusion matrix metrics
cmMetrics <- function(truth, predicted, className) {
    
  # Calculate the confusion matrix
  confusion_matrix <- confusion(truth, predicted)

  # calculate sensitivity
  sens <- sensitivity(confusion_matrix)

  # calculate specificity
  spec <- specificity(confusion_matrix)

  # calculate accuracy
  acc <- accuracy(confusion_matrix)

  # calculate fnr
  fnr <- FNR(confusion_matrix)

  # calculate fpr
  fpr <- FPR(confusion_matrix)

  # put it into a data.frame format
  summ_out <- rbind(sens$byClass, spec$byClass, fnr, fpr, acc$byClass)

  # Only show the metrics with respect to a class of interest
  summ_out <- matrix(summ_out[,className], ncol = 1)

  colnames(summ_out) <- className

  rownames(summ_out) <- c("sensitivity", "specificity",
                          "false negative rate", "false positive rate",
                          "accuracy")

  return(summ_out)
  
} # cmMetrics
