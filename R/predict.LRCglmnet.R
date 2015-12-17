##' Predict (or classify) new data using a fitted glmnet logistic regression classifier
##'
##' @author Landon Sego
##'
##' @importFrom glmnet predict.lognet
##' 
##' @method predict LRCglmnet
##'
##' @param LRCglmnet_object An object of class \code{LRCglmnet}, returned by
##' \code{\link{LRCglmnet}},
##' which contains the optimally-trained elastic net logistic regression classifier
##'
##' @param newdata A dataframe or matrix containing the new set of observations to
##' be predicted, as well as an optional column of true labels.
##' \code{newdata} must contain all of the column names that were used
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
##' An object of class \code{LRCpred} (which inherits
##' from \code{data.frame})
##' that contains the predicted class for each observation.  The columns indicated
##' by \code{truthCol} and \code{keepCols} are included if they were requested.
##'
##'
##' @export
##'
##' @seealso See \code{\link{LRCglmnet}} and \code{\link{LRCglmnet_fit}}
##' for examples.  Also see \code{\link{summary.LRCpred}}.

predict.LRCglmnet <- function(LRCglmnet_object,
                              newdata,
                              truthCol = NULL,
                              keepCols = NULL) {

  # Verify it inherits from the lognet class
  if (!inherits(LRCglmnet_object, "lognet")) {
    stop("Unexpected error:  Object of class 'LRCglmnet' does not inherit from 'lognet'")
  }

  # Switching from column numbers to column names if necessary
  if (!is.null(truthCol) & is.numeric(truthCol)) {
     truthCol <- colnames(newdata)[truthCol]
  }

  if (!is.null(keepCols) & is.numeric(keepCols)) {
     keepCols <- colnames(newdata)[keepCols]
  }

  # Verify the levels of truthCol match the class names in the LRCglmnet_object
  if (!is.null(truthCol)) {

    # It needs to be a factor
    newdata[,truthCol] <- as.factor(newdata[,truthCol])

    if (!setequal(levels(newdata[,truthCol]), LRCglmnet_object$classnames))
      warning("The class labels in the 'truthCol' do not match those ",
              "in the 'LRCglmnet_object'")

  }

  # Get the predictor names expected by the LRCglmnet_object
  predictorNames <- LRCglmnet_object$beta@Dimnames[[1]]

  # Make sure all the predictor names are in the newdata
  if (!all(predictorNames %in% colnames(newdata)))
   stop("The following predictors are expected by 'LRCglmnet_object' but are not\n",
        "present in 'newdata'\n'",
        paste(setdiff(predictorNames, colnames(newdata)), collapse = "', '"), "'\n")

  # Prepare newdata for prediction
  nd <- as.matrix(newdata[,predictorNames])

  if (!is.numeric(nd))
    stop("One or more of the predictor columns in 'newdata' is/are not numeric")

  # Get the original glmnet LRCglmnet_object
  glmnetObject <- LRCglmnet_object[-which(names(LRCglmnet_object) == "optimalParms")]
  class(glmnetObject) <- setdiff(class(LRCglmnet_object), "LRCglmnet")


  # Get the numeric (probability) predictions using predict methods from glmnet package
  # using the optimal lambda
  preds <- predict(glmnetObject, nd,
                   s = LRCglmnet_object$optimalParms["lambda"],
                   type = "response")

  # Dichotomize the prediction using the optimal tau
  predLabels <- factor(preds > LRCglmnet_object$optimalParms["tau"],
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

    class(output) <- c("LRCpred", class(output))

    attributes(output) <- c(attributes(output),
                            list(modelType = "glmnet",
                                 truthCol = truthCol,
                                 optimalParms = LRCglmnet_object$optimalParms,
                                 classNames = glmnetObject$classnames))


  }

  else {
    attributes(output) <- c(attributes(output),
                            list(modelType = "glmnet",
                                 optimalParms = LRCglmnet_object$optimalParms,
                                 classNames = glmnetObject$classnames))
  }

  return(output)

} # predict.LRCglmnet
