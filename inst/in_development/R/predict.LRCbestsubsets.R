##' Predict (or classify) new data using a fitted best subsets logistic regression
##' classifier
##'
##' @author Landon Sego
##'
##' @method predict LRCbestsubsets
##'
##' @param LRCbestsubsets_object An object of class \code{LRCbestsubsets}, returned by
##' \code{\link{LRCbestsubsets}},
##' which contains the best subsets logistic regression classifier with optimal threshold
##' \eqn{\tau}.
##'
##' @param newdata A dataframe containing the new set of observations to
##' be predicted, as well as an optional column of true labels.
##' \code{newdata} must contain all of the column names that were used
##' to fit best subsets logistic regression classifier.
##'
##' @param truthCol The column number or column name in \code{newdata} that contains the
##' true labels. Optional.
##'
##' @param keepCols A numeric vector of column numbers (or a character vector of
##' column names) that will be 'kept' and returned with the predictions. Optional.
##'
##'
##' @return
##' \item{\code{predict.LRCbestsubsets}}{
##' An object of class \code{LRCbestsubsets_pred} (which inherits
##' from \code{data.frame})
##' that contains the predicted class for each observation.  The columns indicated
##' by \code{truthCol} and \code{keepCols} are included if they were requested.}
##'
##' @export
##' 
##' @seealso See \code{\link{LRCbestsubsets}} and \code{\link{LRCbestsubsets_fit}}
##' for examples.  Also see \code{\link{summary.LRCpred}}.

predict.LRCbestsubsets <- function(LRCbestsubsets_object,
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


  # Verify the levels of truthCol match the class names in the LRCbestsubsets_object
  if (!is.null(truthCol)) {

    # It needs to be a factor
    newdata[,truthCol] <- as.factor(newdata[,truthCol])

    if (!setequal(levels(newdata[,truthCol]), LRCbestsubsets_object$classnames))
      warning("The class labels in the 'truthCol' do not match those ",
              "in the 'LRCbestsubsets_object'")

  }

  # Get the predictor names expected by the LRCbestsubsets_object
  predictorNames <- setdiff(names(coef(LRCbestsubsets_object)), "(Intercept)")

  # Make sure all the predictor names are in the newdata
  if (!all(predictorNames %in% colnames(newdata)))
   stop("The following predictors are expected by 'LRCbestsubsets_object' but are not\n",
        "present in 'newdata'\n'",
        paste(setdiff(predictorNames, colnames(newdata)), collapse = "', '"), "'\n")

  # Prepare newdata for prediction (select and order predictors)
  # The 'as.data.frame' preserves the dataframe even if only one predictor is
  # selected
  nd <- as.data.frame(newdata[,predictorNames])
  colnames(nd) <- predictorNames

  # Get the original glm LRCbestsubsets_object
##   glmObject <- LRCbestsubsets_object[-which(names(LRCbestsubsets_object) %in%
##                                             c("tau", "optimalTaus", "classnames"))]
  glmObject <- LRCbestsubsets_object
  class(glmObject) <- setdiff(class(LRCbestsubsets_object), "LRCbestsubsets")

  # Get the numeric (probability) predictions from predict.glm using the optimal lambda
  preds <- predict(glmObject, nd, type = "response")

  # Dichotomize the prediction using the optimal tau
  predLabels <- factor(preds > LRCbestsubsets_object$tau,
                       levels = c(FALSE, TRUE),
                       labels = LRCbestsubsets_object$classnames)

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
                            list(modelType = "glm",
                                 truthCol = truthCol,
                                 optimalParms = LRCbestsubsets_object$tau,
                                 classNames = LRCbestsubsets_object$classnames))


  }

  else {
    attributes(output) <- c(attributes(output),
                            list(modelType = "glm",
                                 optimalParms = LRCbestsubsets_object$tau,
                                 classNames = LRCbestsubsets_object$classnames))
  }

  return(output)

} # predict.LRCbestsubsets


