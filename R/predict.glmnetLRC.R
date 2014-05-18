##' Predict (or classify) new data using a fitted glmnet logistic regression classifier
##'
##' @author Landon Sego
##'
##'  
##' @method predict glmnetLRC
##' 
##' @param glmnetLRCobject An object of class \code{glmnetLRC}, returned by
##' \code{\link{glmnetLRC}},
##' which contains the optimally-trained elastic net logistic regression classifier
##' 
##' @param newdata A dataframe or matrix containing the new set of observations to
##' be predicted. \code{newdata} must contain all of the column names that were used
##' to fit the elastic net logistic regression classifier.
##' 
##' @param truthCol The column number or column name in \code{newdata} that contains the
##' true labels. Optional.
##' 
##' @param keepCols A numeric vector of column numbers (or a character vector of
##' column names) that will be 'kept' and returned with the predictions. Optional.
##'
##'
##' @return
##' \describe{
##' \item{\code{predict.glmnetLRC}}{
##' An object of class \code{glmnetLRCpred} (which inherits
##' from \code{data.frame}
##' that contains the predicted class for each observation.  The columns indicated
##' by \code{truthCol} and \code{keepCols} are included if they were requested.}
##'
##' \item{\code{summary.glmnetLRCpred}}{
##'
##' If \code{truthCol} was provided in the call to
##' \code{predict.glmnetLRC}, a
##' \code{data.frame} is returned with the sensitivity, specificity, false negative rate,
##' false positive rate, and accuracy for the class designated by the second level of
##' the \code{truthLabels} argument in \code{\link{glmnetLRC}}.
##'
##' If \code{truthCol = NULL} in the call to \code{predict.glmnetLRC}, a tabulation
##' of the number of predictions for each class is returned.
##' }
##' }
##' 
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} for examples

predict.glmnetLRC <- function(glmnetLRCobject,
                              newdata,
                              truthCol = NULL,
                              keepCols = NULL) {

  # Switching from column numbers to column names if necessary
  if (!is.null(truthCol) & is.numeric(truthCol)) {
     truthCol <- colnames(newdata)[truthCol]
  }

  if (!is.null(keepCols) & is.numeric(keepCols)) {
     keepCols <- colnames(newdata)[keepCols]
  }


  # Verify the levels of truthCol match the class names in the glmnetLRCobject
  if (!is.null(truthCol)) {

    # It needs to be a factor
    newdata[,truthCol] <- as.factor(newdata[,truthCol])

    if (!setequal(levels(newdata[,truthCol]), glmnetLRCobject$classnames))
      warning("The class labels in the 'truthCol' do not match those ",
              "in the 'glmnetLRCobject'")

  }

  # Get the predictor names expected by the glmnetLRCobject
  predictorNames <- glmnetLRCobject$beta@Dimnames[[1]]

  # Make sure all the predictor names are in the newdata
  if (!all(predictorNames %in% colnames(newdata)))
   stop("The following predictors are expected by 'glmnetLRCobject' but are not\n",
        "present in 'newdata'\n'",
        paste(setdiff(predictorNames, colnames(newdata)), collapse = "', '"), "'\n")

  # Prepare newdata for prediction
  nd <- as.matrix(newdata[,predictorNames])
  
  if (!is.numeric(nd))
    stop("One or more of the predictor columns in 'newdata' is/are not numeric")
  
  # Get the original glmnet glmnetLRCobject
  glmnetObject <- glmnetLRCobject[-which(names(glmnetLRCobject) == "optimalParms")]
  class(glmnetObject) <- setdiff(class(glmnetLRCobject), "glmnetLRC")

  
  # Get the numeric (probability) predictions from predict.glmnet using the optimal lambda
  preds <- predict(glmnetObject, nd,
                   s = glmnetLRCobject$optimalParms["lambda"],
                   type = "response")

  # Dichotomize the prediction using the optimal tau    
  predLabels <- factor(preds > glmnetLRCobject$optimalParms["tau"],
                       levels = c(FALSE, TRUE),
                       labels = glmnetObject$classnames)

  # If there were rownames in newdata, add them in
  if (!is.null(rn <- rownames(newdata)))
    names(predLabels) <- rn

  # Combine new data
  output <- cbind(predLabels, newdata[,truthCol], newdata[,keepCols])
  colnames(output) <- c("PredictClass", truthCol, keepCols)
  
  # Assign the class if a truth column was provided
  if (!is.null(truthCol)) {

    class(output) <- c("glmnetLRCpred", class(output))
    
    attributes(output) <- c(attributes(output),
                            list(truthCol = truthCol,
                                 optimalParms = glmnetLRCobject$optimalParms,
                                 classNames = glmnetObject$classnames))


  }

  else {
    attributes(output) <- c(attributes(output),
                            list(optimalParms = glmnetLRCobject$optimalParms,
                                 classNames = glmnetObject$classnames))
  }

  return(output)

} # predict.glmnetLRC


##' @rdname predict.glmnetLRC
##' 
##' @method summary glmnetLRCpred
##' 
##' @param glmnetLRCpredObject an object of class \code{glmnetLRCpred} returned by
##' \code{predict.glmnetLRC}.
##' 
##' @export


summary.glmnetLRCpred <- function(glmnetLRCpredObject) {

  truthCol <- attributes(glmnetLRCpredObject)$truthCol
  
  # If there are any missing data in the PredictClass or the Truth class,
  # remove them
  cc <- complete.cases(glmnetLRCpredObject[, c("PredictClass", truthCol)])

  if (any(!cc)) {
    
    warning("'glmnetLRCpredObject' has ", sum(!cc),
            " missing observation(s) which will be removed\n")

    glmnetLRCpredObject <- glmnetLRCpredObject[cc,]

  }
  
  # Calculate the confusion matrix  
  confusion_matrix <- confusion(glmnetLRCpredObject[, truthCol],
                                glmnetLRCpredObject[,"PredictClass"])
  
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
  summ_out <- matrix(summ_out[,attributes(glmnetLRCpredObject)$classNames[2]], ncol = 1)

  colnames(summ_out) <- attributes(glmnetLRCpredObject)$classNames[2]
  
  rownames(summ_out) <- c("sensitivity", "specificity",
                          "false negative rate", "false positive rate",
                          "accuracy")
  
  return(as.data.frame(summ_out))
  
} # end summary.glmnetLRCpred
