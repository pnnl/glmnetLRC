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
