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
