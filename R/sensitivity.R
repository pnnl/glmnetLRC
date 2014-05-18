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
