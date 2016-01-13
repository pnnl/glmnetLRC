# THIS IS SQM package stuff--eventually, I'll want to call these functions with the SQM
# package

## Computes the confusion summary for a vector of classifications and a ground
## truth vector.
##
## For a vector of classifications and truth labels,a confusion matrix is 
## generated. Binary and multi-class classifications are permitted and 
## following four measures are computed for each class:
##
##  \itemize{
##  \item True Positives (TP)
##  \item True Negatives (TN)
##  \item False Positives (FP)
##  \item False Negatives (FN)
##  }
## 
## For multi-class classification, each class is considered in a binary context.
## For example, suppose that we have the three food condiment classes: ketchup,
## mustard, and other. When calculating the TP, TN, FP, and FN values for
## ketchup, we consider each observation as either 'ketchup' or 'not ketchup.'
## Similarly, for mustard, we would consider 'mustard' and 'not mustard', and for
## other, we would consider 'other' and 'not other.'
##
## With the above counts for each class, we can quickly calculate a variety
## of class-specific and aggregate classification accuracy measures.
##
## @export
## 
## @rdname confusion
## @author John Ramey
## 
## @param truthClass vector of ground truth classification labels
## @param predictedClass vector of predicted classification labels
## @return list with the results of confusion matrix results for each class.
## 
## @examples
## 
## # load the test data and the trained lasso model   
## data(testdata)
## data(lassoModel)
## 
## # Make predictions from the test data
## predictTrain <- predict(lassoModel, traindata,
##                         truthCol = "Curated_Quality",
##                         keepCols = 12:14)
##                         
## # Remove all NA cases produced in the prediction phase
## indexNA <- !is.na(predictTrain$PredictClass)
## cleanPredict <- predictTrain[indexNA,]
## 
## # Create the confusion matrix
## confmat <- confusion(cleanPredict[,"Curated_Quality"], cleanPredict[,"PredictClass"])

confusion <- function(truthClass, predictedClass) {
  
  truthClass <- factor(truthClass)
  
  if(!all(unique(predictedClass) %in% levels(truthClass))) {
    stop("The vector of predicted classes contains classes that are not present
          in the vector of truth classes.")
  } else {
    predictedClass <- factor(predictedClass, levels = levels(truthClass))
  }
  
  # The confusion matrix contains a summary of the correct and incorrect
  # classifications by class.
  confusionMatrix <- table(predictedClass, truthClass)
  
  # For each class, we compute the true positives, true negatives,
  # false positives, and false negatives from the confusion matrix.
  classSummary <- lapply(seq_len(nlevels(truthClass)), function(i) {
    classSummary <- list()
    classSummary$truePos <- confusionMatrix[i, i]
    classSummary$trueNeg <- sum(confusionMatrix[-i, -i])
    classSummary$falsePos <- sum(confusionMatrix[i, -i])
    classSummary$falseNeg <- sum(confusionMatrix[-i, i])
    classSummary$classSampleSize <- sum(confusionMatrix[,i])
    classSummary
  })
  names(classSummary) <- levels(truthClass)

  # We return to the calling code the summary for each class,
  # the total sample size, the number of classes
  confusionSummary <- list()
  confusionSummary$classSummary <- classSummary
  confusionSummary$sampleSize <- length(truthClass)
  confusionSummary$numClasses <- nlevels(truthClass)

  confusionSummary
}


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

## Computes classification accuracy from the confusion matrix summary based on a
## set of predicted and truth classes for a signature.
##
## For each class, we calculate the classification accuracy in order to summarize
## its performance for the signature. We compute one of two aggregate scores,
## to summarize the overall performance of the signature.
## 
## We define the accuracy as the proportion of correct classifications.
##
## The two aggregate score options are the macro- and micro-aggregate (average)
## scores. The macro-aggregate score is the arithmetic mean of the binary scores
## for each class. The micro-aggregate score is a weighted average of each class'
## binary score, where the weights are determined by the sample sizes for each
## class. By default, we use the micro-aggregate score because it is more robust,
## but the macro-aggregate score might be more intuitive to some users.
##
## Note that the macro- and micro-aggregate scores are the same for classification
## accuracy.
##
## The accuracy measure ranges from 0 to 1 with 1 being the optimal value.
##
## @export
## 
## @rdname accuracy
## 
## @param confusionSummary list containing the confusion summary for a set of
## classifications
## 
## @param aggregate string that indicates the type of aggregation; by default,
## micro. See details.
## 
## @return list with the accuracy measure for each class as well as the macro-
## and micro-averages (aggregate measures across all classes).
## 
## @examples 
## # load the test data and the trained lasso model   
## data(testdata)
## data(lassoModel)
## 
## # Make predictions from the test data
## predictTrain <- predict(lassoModel, traindata,
##                         truthCol = "Curated_Quality",
##                         keepCols = 12:14)
##                         
## # Remove all NA cases produced in the prediction phase
## indexNA <- !is.na(predictTrain$PredictClass)
## cleanPredict <- predictTrain[indexNA,]
## 
## # Create the confusion matrix
## confmat <- confusion(cleanPredict[,"Curated_Quality"], cleanPredict[,"PredictClass"])
## 
## # Compute the accuracy
## accuracy(confmat)
## 

accuracy <- function(confusionSummary, aggregate = c('micro', 'macro')) {
  aggregate <- match.arg(aggregate)
  
  byClass <- sapply(confusionSummary$classSummary, function(clSummary) {
    with(clSummary,
         (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
    )
  })
  names(byClass) <- names(confusionSummary$classSummary)

  if (aggregate == 'micro') {
    numerator <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
	with(clSummary, truePos + trueNeg)
    }))
    denom <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
	with(clSummary, truePos + trueNeg + falsePos + falseNeg)
    }))
    aggregate <- numerator/denom
  } else {
    aggregate <- mean(byClass)
  }

  list(byClass = byClass, aggregate = aggregate)
}


## Computes classification sensitivity from the confusion matrix summary based on
## a set of predicted and truth classes for a signature.
##
## For each class, the classification sensitivity is computed to
## summarize its performance for the signature. It is possible to compute 
## one of two aggregate scores, to summarize the overall performance of the signature.
##
## Suppose that an observation can be classified into one
## (and only one) of K classes. For the jth class (j = 1, ..., K),
## the sensitivity is defined as the conditional probability
##
## Sensitivity_j = Pr(y_hat = j | y = j),
##
## where y_hat and y are the empirical and true classifications,
## respectively.
##
## The estimate of sensitivity_j for the jth class is
##
## (TP_j) / (TP_j + FN_j),
##
## where TP_j and FN_j are the true positives and false negatives,
## respectively. More specifically, TP_j is the number of observations
## that are correctly classified into the jth class, and FN_j
## is the number of observations that, in truth, belong to class j but 
## were classified incorrectly.
##
## The two aggregate score options are the macro- and micro-aggregate (average)
## scores. The macro-aggregate score is the arithmetic mean of the binary scores
## for each class. The micro-aggregate score is a weighted average of each class'
## binary score, where the weights are determined by the sample sizes for each
## class. By default, the micro-aggregate score is used because it is more robust,
## but the macro-aggregate score might be more intuitive to some users.
## 
## In statistical terms, notice that in the binary case (K = 2), the sensitivity
## is the recall.
##
## Also, note that the sensitivity is equal to the TPR.
##
## The sensitivity measure ranges from 0 to 1 with 1 being the optimal value.
##
## @export
## 
## @rdname sensitivity
## 
## @param confusionSummary list containing the confusion summary for a set of
## classifications
## @param aggregate string that indicates the type of aggregation; by default,
## micro. See details.
## @return list with the accuracy measure for each class as well as the macro-
## and micro-averages (aggregate measures across all classes).
## 
## @examples
## 
## # load the test data and the trained lasso model   
## data(testdata)
## data(lassoModel)
## 
## # Make predictions from the test data
## predictTrain <- predict(lassoModel, traindata,
##                         truthCol = "Curated_Quality",
##                         keepCols = 12:14)
##                         
## # Remove all NA cases produced in the prediction phase
## indexNA <- !is.na(predictTrain$PredictClass)
## cleanPredict <- predictTrain[indexNA,]
## 
## # Create the confusion matrix
## confmat <- confusion(cleanPredict[,"Curated_Quality"], cleanPredict[,"PredictClass"])
## 
## # Compute the specificity for classes 'good' and 'poor'
## sensitivity(confmat)


sensitivity <- function(confusionSummary, aggregate = c('micro', 'macro')) {
  aggregate <- match.arg(aggregate)
  
  byClass <- sapply(confusionSummary$classSummary, function(clSummary) {
    with(clSummary,
         truePos / (truePos + falseNeg)
    )
  })
  names(byClass) <- names(confusionSummary$classSummary)

  if (aggregate == 'micro') {
    numerator <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
      clSummary$truePos
    }))
    denom <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
      with(clSummary, truePos + falseNeg)
    }))
    aggregate <- numerator / denom
  } else {
    aggregate <- mean(byClass)
  }

  list(byClass = byClass, aggregate = aggregate)
}

## Computes classification specificity from the confusion matrix summary based on
## a set of predicted and truth classes for a signature.
##
## For each class, the classification specificity is computed to
## summarize its performance for the signature. One of two aggregate
## scores are computed to summarize the overall performance of the signature.
##
## The estimated specificity for class j is 
##
## (TN_j) / (TN_j + FP_j),
##
## where TN_j and FP_j are the true negatives and false positives,
## respectively. More specifically, TN_j is the number of observations
## that are correctly classified into other classes than the jth class, and FP_j
## is the number of observations that are incorrectly classified into class
## j.
##
## The two aggregate score options are the macro- and micro-aggregate (average)
## scores. The macro-aggregate score is the arithmetic mean of the binary scores
## for each class. The micro-aggregate score is a weighted average of each class'
## binary score, where the weights are determined by the sample sizes for each
## class. By default, the micro-aggregate score is used because it is more robust,
## but the macro-aggregate score might be more intuitive to some users.
##
## Notice that the specificity is equal to the TNR.
##
## The specificity measure ranges from 0 to 1 with 1 being the optimal value.
##
## @export
## 
## @rdname specificity
## 
## @param confusionSummary list containing the confusion summary for a set of
## classifications
## @param aggregate string that indicates the type of aggregation; by default,
## micro. See details.
## @return list with the accuracy measure for each class as well as the macro-
## and micro-averages (aggregate measures across all classes).
## 
## @examples
## 
## # load the test data and the trained lasso model   
## data(testdata)
## data(lassoModel)
## 
## # Make predictions from the test data
## predictTrain <- predict(lassoModel, traindata,
##                         truthCol = "Curated_Quality",
##                         keepCols = 12:14)
##                         
## # Remove all NA cases produced in the prediction phase
## indexNA <- !is.na(predictTrain$PredictClass)
## cleanPredict <- predictTrain[indexNA,]
## 
## # Create the confusion matrix
## confmat <- confusion(cleanPredict[,"Curated_Quality"], cleanPredict[,"PredictClass"])
## 
## # Compute the specificity for classes 'good' and 'poor'
## specificity(confmat)
## 
 
specificity <- function(confusionSummary, aggregate = c('micro', 'macro')) {
  aggregate <- match.arg(aggregate)
  
  byClass <- sapply(confusionSummary$classSummary, function(clSummary) {
    with(clSummary,
         trueNeg / (trueNeg + falsePos)
    )
  })
  names(byClass) <- names(confusionSummary$classSummary)

  if (aggregate == 'micro') {
    numerator <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
      clSummary$trueNeg
    }))
    denom <- sum(sapply(confusionSummary$classSummary, function(clSummary) {
      with(clSummary, trueNeg + falsePos)
    }))
    aggregate <- numerator / denom
  } else {
    aggregate <- mean(byClass)
  }

  list(byClass = byClass, aggregate = aggregate)
}

