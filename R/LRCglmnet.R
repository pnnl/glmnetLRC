##' Elastic net logistic regression classifier (LRC) with arbitrary loss function
##'
##' This functions extends the \code{\link{glmnet}} and \code{\link{cv.glmnet}}
##' functions from the \pkg{glmnet}
##' package. It uses cross validation to identify optimal elastic net parameters and a
##' threshold parameter for binary classification, where optimality is defined
##' by minimizing an arbitrary, user-specified discrete loss function.
##'
##' @details
##' For a given partition of the training data, cross validation is
##' performed to estimate the optimal values of
##' \eqn{\alpha} (the mixing parameter of the L1 and L2 norms) and \eqn{\lambda}
##' (the regularization parameter), as well as the optimal threshold, \eqn{\tau},
##' which is used to dichotomize the probability predictions of the elastic net
##' logistic regression model into binary outcomes.
##' (Specifically, if the probability an observation
##' belongs to the second level of \code{truthLabels} exceeds \eqn{\tau}, it is
##' classified as belonging to that second level).  In this case, optimality is defined
##' as the set of parameters that minimize the risk, or expected loss, where the
##' loss function is defined by \code{lossMat}.  The expected loss is calculated such
##' that each observation in the data receives equal weight, following equation 7.49
##' from Hastie et al.
##'
##' \code{LRCglmnet} searches for the optimal values of \eqn{\alpha} and \eqn{\tau} by
##' fitting the elastic net model at the points of the two-dimensional grid defined by
##' \code{alphaVec} and \code{tauVec}.  For each of these points, the vector of
##' \eqn{\lambda}
##' values is selected automatically by \code{\link{glmnet}} according to its default
##' arguments.  Note that the \eqn{\lambda} vector depends on \eqn{\alpha}.
##' The expected loss is calculated for each \eqn{(\alpha,\lambda,\tau)} triple, and the
##' triple giving rise to the lowest risk designates the optimal model for a given
##' cross-validation partition of the data.
##'
##' This process is repeated \code{cvReps} times, where each time a different random
##' partition of the data is created using its own seed, resulting in another
##' 'optimal' estimate of \eqn{(\alpha,\lambda,\tau)}.  The final estimate of
##' \eqn{(\alpha,\lambda,\tau)} is given by the respective medians of those estimates.
##' The final elastic net logistic regression classfier is given by fitting the regression
##' coefficients to all the training data using the optimal \eqn{(\alpha,\lambda,\tau)}.
##'
##' In general, the methodology discussed here follows the Appendix material found in
##' Amidan et al.
##'
##' @author Landon Sego, Alex Venzin
##'
##' @rdname LRCglmnet
##'
##' @export
##'
##' @param truthLabels A factor with two levels containing the true labels for each
##' observation.
##' If it is more desirable to correctly predict one of the two classes over the other,
##' the second level of this factor should be the class you are most interested in
##' predicting correctly.
##'
##' @param predictors A matrix whose columns are the explanatory regression variables.  Note:
##' factors are not currently supported.  To include a factor variable with n levels, it must be represented
##' as n-1 dummy variables in the matrix.
##'
##' @param lossMat A loss matrix of class \code{lossMat}, produced by
##' \code{\link{lossMatrix}}, that specifies the penalties for classification errors.
##'
##' @param weight The observation weights that are passed to \code{\link{glmnet}}.
##' The default value
##' is 1 for each observation. Refer to \code{\link{glmnet}} for further
##' information.
##'
##' @param alphaVec A vector of potential values for the elastic net mixing parameter,
##' \eqn{\alpha}. A value of \eqn{\alpha = 1} is the lasso penalty, \eqn{\alpha = 0} is the ridge penalty.
##' Refer to \code{\link{glmnet}} for further information.
##'
##' @param tauVec A sequence of \eqn{\tau} threshold values in (0, 1) for the
##' logistic regression classifier. For a new observation, if the predicted probability
##' that the observation belongs to the second level
##' of \code{truthLabels} exceeds tau, the observation is classified as belonging
##' to the second level.
##'
##' @param intercept A logical indicating whether an intercept should be included in glmnet model,
##' passed to the \code{intercept} argument of \code{\link{glmnet}}.
##'
##' @param naFilter The maximum proportion of data observations that can be missing
##' for any single column in \code{predictors}. If, for a given predictor, the proportion of
##' sample observations which are NA is greater than \code{naFilter}, the predictor is
##' not included in the elastic net model fitting.
##'
##' @param cvFolds The number of cross validation folds.
##\code{cvFolds = NROW(predictors)}  <-- NEED to work on this
## gives leave-one-out cross validation.
##'
##' @param cvReps The number of cross validation replicates, i.e., the number
##' of times to repeat the cross validation
##' by randomly repartitioning the data into folds.
##'
##' @param masterSeed The random seed used to generate unique seeds for
##' each cross validation replicate.
##'
##' @param nJobs The number of cores on the local host
##' to use in parallelizing the training.  Parallelization
##' takes place at the \code{cvReps} level, i.e., if \code{cvReps = 1}, parallelizing would
##' do no good, whereas if \code{cvReps = 2}, each rep would be run separately in its
##' own thread if \code{nJobs = 2}.
##' Parallelization is executed using \code{\link{parLapplyW}} from the
##' \pkg{Smisc} package.
##'
##' @param estimateLoss A logical, set to \code{TRUE} to calculate the average loss estimated via
##' cross validation using the optimized parameters \eqn{(\alpha, \lambda, \tau)} to fit the elastic
##' net model for each cross validation fold.
##' This requires another cross-validation pass through the data, but using only
##' the optimal parameters to estimate the loss for each cross-validation replicate.
##'
##' @param verbose A logical, set to \code{TRUE} to receive messages regarding
##' the progress of the training algorithm.
##'
##' @return
##' Returns an object of class \code{LRCglmnet}, which
##' inherits from classes \code{lognet} and \code{glmnet}.  It contains the
##' object returned by \code{\link{glmnet}} that has been fit to all the data using
##' the optimal parameters \eqn{(\alpha, \lambda, \tau)}.
##' It also contains the following additional elements:
##' \describe{
##' \item{parms}{Contains the parameter estimates for \eqn{(\alpha, \lambda, \tau)} that minmize the expected
##' loss for each cross validation replicate.  Used by the \code{plot} method.}
##' \item{optimalParms}{Contains the final estimates of \eqn{(\alpha, \lambda, \tau)}, calculated as the
##' element-wise median of \code{parms}}
##' \item{lossEstimates}{If \code{estimateLoss = TRUE}, this element contains the expected loss
##' for each cross validation replicate}
##' }
##'
##' @references
##' Amidan BG, Orton DJ, LaMarche BL, Monroe ME, Moore RJ,
##' Venzin AM, Smith RD, Sego LH, Tardiff MF, Payne SH. 2014.
##' Signatures for Mass Spectrometry Data Quality.
##' Journal of Proteome Research. 13(4), 2215-2222.
##' \url{http://pubs.acs.org/doi/abs/10.1021/pr401143e}
##'
##' Hastie T, Tibshirani R, Friedman JH. 2008. The Elements of Statistical Learning:
##' Data Mining, Inference, and Prediction. 2nd edition. Springer-Verlag.
##'
##' @seealso \code{\link{summary.LRCpred}}, a summary method for objects of class
##' \code{LRCpred}, produced by the \code{predict} method.
##' 
##' @examples
##' # Load the VOrbitrap Shewanella QC data from Amidan et al.
##' data(traindata)
##'
##' # Here we select the predictor variables
##' predictors <- as.matrix(traindata[,9:96])
##'
##' # The logistic regression model requires a binary response
##' # variable. We will create a factor variable from the
##' # Curated Quality measurements. Note how we put "poor" as the
##' # second level in the factor.  This is because the principal
##' # objective of the classifer is to detect "poor" datasets
##' response <- factor(traindata$Curated_Quality,
##'                    levels = c("good", "poor"),
##'                    labels = c("good", "poor"))
##'
##' # Specify the loss matrix. The "poor" class is the target of interest.
##' # The penalty for misclassifying a "poor" item as "good" results in a
##' # loss of 5.
##' lM <- lossMatrix(c("good","good","poor","poor"),
##'                  c("good","poor","good","poor"),
##'                  c(     0,     1,     5,     0))
##'
##' # Display the loss matrix
##' lM
##'
##' # Train the elastic net classifier (we don't run it here because it takes a long time)
##' \dontrun{
##' loadNamespace("parallel")
##' LRCglmnet_fit <- LRCglmnet(response, predictors, lM, nJobs = parallel::detectCores())
##' }
##'
##' # We'll load the precalculated model fit instead
##' data(LRCglmnet_fit)
##' 
##' # Show the optimal parameter values
##' print(LRCglmnet_fit)
##'
##' # Show the plot of all the optimal parameter values for each cross validation replicate
##' plot(LRCglmnet_fit)
##'
##' # Load the new observations
##' data(testdata)
##'
##' # Use the trained model to make predictions about
##' # new observations for the response variable.
##' new <- predict(LRCglmnet_fit, testdata, truthCol = "Curated_Quality", keepCols = 1:2)
##' head(new)
##'
##' # Now summarize the performance of the model
##' summary(new)
##'
##' # If predictions are made without the an indication of the ground truth,
##' # the summary is simpler:
##' summary(predict(LRCglmnet_fit, testdata))

LRCglmnet <- function(truthLabels, predictors, lossMat,
                      weight = rep(1, NROW(predictors)),
                      alphaVec = seq(0, 1, by = 0.2),
                      tauVec = seq(0.1, 0.9, by = 0.05),
                      intercept = TRUE,
                      naFilter = 0.6,
                      cvFolds = 5,
                      cvReps = 100,
                      masterSeed = 1,
                      nJobs = 1,
                      estimateLoss = FALSE,
                      verbose = FALSE) {

  # Check inputs
  stopifnot(is.factor(truthLabels),
            length(levels(truthLabels)) == 2,
            NCOL(predictors) > 1,
            is.matrix(predictors),
            is.numeric(predictors),
            length(truthLabels) == NROW(predictors),
            inherits(lossMat, "lossMat"),
            is.numeric(alphaVec),
            all(alphaVec <= 1) & all(alphaVec >= 0),
            is.numeric(tauVec),
            all(tauVec < 1) & all(tauVec > 0),
            length(intercept) == 1,
            is.logical(intercept),
            length(naFilter) == 1,
            is.numeric(naFilter),
            (naFilter < 1) & (naFilter > 0),
            length(cvFolds) == 1,
            is.numeric(cvFolds),
            cvFolds %% 1 == 0,
            cvFolds >= 2,
            cvFolds <= NROW(predictors),
            length(cvReps) == 1,
            is.numeric(cvReps),
            cvReps %% 1 == 0,
            cvReps > 0,
            is.numeric(masterSeed),
            length(masterSeed) == 1,
            is.numeric(nJobs),
            nJobs >= 1,
            is.logical(estimateLoss),
            length(estimateLoss) == 1,
            is.logical(verbose),
            length(verbose) == 1)

  # Force the evaluation of the weight object immediately--this is IMPORTANT
  # because of R's lazy evaluation
  force(weight)

  ################################################################################
  # Data preparation
  ################################################################################

  # Get the data and filter missing values as neccessary
  d <- dataPrep(truthLabels, predictors, weight, naFilter, verbose)

  # number of observations after filtering for NAs
  n <- length(d$truthLabels)

  # Report the number of observations
  if (verbose) {
    cat(n, "observations are available for fitting the LRCglmnet model\n")
  }

  # A wrapper function for calling single_LRCglmnet() via Smisc::parLapplyW()
  trainWrapper <- function(seed,
                           alphaVec = 1,
                           tauVec = 0.5,
                           lambdaVal = NULL,
                           lambdaVec = NULL) {

    single_LRCglmnet(d$truthLabels,
                     d$predictors,
                     lossMat,
                     d$weight,
                     alphaVec,
                     tauVec,
                     intercept,
                     cvFolds,
                     seed,
                     n,
                     verbose,
                     lambdaVal = lambdaVal,
                     lambdaVec = lambdaVec)

  } # trainWrapper

  # Get the vector of seeds that will ensure repeatability across threads
  seedVec <- createSeeds(masterSeed, cvReps)

  ################################################################################
  # Run in parallel
  ################################################################################
  if (nJobs > 1) {

    # Object names that will be needed in the cluster
    neededObjects <- c("d",
                       "alphaVec",
                       "tauVec",
                       "n",
                       "cvFolds",
                       "lossMat",
                       "verbose")


    # Execute the training in parallel
    pe <- Smisc::parLapplyW(seedVec, trainWrapper,
                            alphaVec = alphaVec,
                            tauVec = tauVec,
                            njobs = nJobs,
                            expr = expression(library(lrc)),
                            varlist = neededObjects)

    # Collapse results to a data frame
    parmEstimates <- Smisc::list2df(pe, row.names = 1:cvReps)

  }

  ################################################################################
  # Single thread
  ################################################################################

  else {

    parmEstimates <- Smisc::list2df(lapply(seedVec, trainWrapper,
                                           alphaVec = alphaVec,
                                           tauVec = tauVec),
                                    row.names = 1:cvReps)

  }

  # The final parameter estimates will be the (alpha, lambda, tau) centroid in
  # the parameter estimates

  # Use the median instead...
  finalParmEstimates <- apply(parmEstimates[,c("alpha", "lambda", "tau")], 2, median)


  ################################################################################
  # Create the final model from the averaged parameters
  ################################################################################

  # Create an aggregated lambda sequence to fit the final lasso logisitc
  # regression model. Per the documentation in glmnetfit, apparently it does better
  # with a sequence of lambdas during the fitting.  But the predict method only
  # operates on the optimal lambda
  sdLambda <- sd(parmEstimates$lambda)
  lambdaVec <- sort(c(finalParmEstimates[["lambda"]],
                      seq(finalParmEstimates[["lambda"]] + sdLambda,
                          max(finalParmEstimates[["lambda"]] - sdLambda, 1e-04),
                          length = 50)),
                      decreasing = TRUE)

  # Fit the model using the aggregate parameters
  glmnetFinal <- glmnet::glmnet(d$predictors, d$truthLabels, weights = d$weight,
                                family = "binomial", lambda = lambdaVec,
                                alpha = finalParmEstimates[["alpha"]],
                                intercept = intercept)

  # Return the optimal parameters for graphical output
  glmnetFinal$parms <- parmEstimates

  # Return the aggregated optimal parameters (gridVals)
  glmnetFinal$optimalParms <- finalParmEstimates

  ################################################################################
  # If we are to estimate the loss by averaging the cross validation loss over
  # all the CV replicates, using the optimal parameters for each fit to a CV
  # training set.  Will use the same random seeds as before to constructe
  # the training/testing sets.
  ################################################################################

  if (estimateLoss) {

    ################################################################################
    # Run on the cluster which was set up earlier
    ################################################################################
    if (nJobs > 1) {

      # Execute the loss calculation in parallel
      le <- Smisc::parLapplyW(seedVec, trainWrapper,
                              alphaVec = finalParmEstimates[["alpha"]],
                              tauVec = finalParmEstimates[["tau"]],
                              lambdaVec = lambdaVec,
                              lambdaVal = finalParmEstimates[["lambda"]],
                              expr = expression(library(lrc)),
                              varlist = neededObjects,
                              njobs = nJobs)

      # Collapse results to a data frame
      lossEstimates <- Smisc::list2df(le, row.names = 1:cvReps)

      
    }

    ################################################################################
    # Single thread
    ################################################################################

    else {

      lossEstimates <-
        Smisc::list2df(lapply(seedVec, trainWrapper,
                              alphaVec = finalParmEstimates[["alpha"]],
                              tauVec = finalParmEstimates[["tau"]],
                              lambdaVec = lambdaVec,
                              lambdaVal = finalParmEstimates[["lambda"]]),
                       row.names = 1:cvReps)
    }


    # Include the loss estimates in the returned object
    glmnetFinal$lossEstimates <- lossEstimates

  } # finish estimating loss

  # Assign the class
  class(glmnetFinal) <- c("LRCglmnet", class(glmnetFinal))


  return(glmnetFinal)

} # LRCglmnet

##' @method print LRCglmnet
##'
##' @describeIn LRCglmnet Displays the overall optimized values of
##' \eqn{(\alpha, \lambda, \tau)}, with the corresponding degrees of freedom and
##' deviance for the model fit to all the data using the optimzed parameters.  If \code{estimateLoss = TRUE}
##' when \code{LRCglmnet} was called, the mean and standard deviation of the expected loss are also shown.
##' In addition, all of this same information is returned invisibly as a matrix.
##'
##' @param x For the \code{print} and \code{plot} methods:  an object of class \code{LRCglmnet}, returned
##' by \code{LRCglmnet}, which contains the optimally-trained elastic net logistic regression classifier
##'
##' @export

print.LRCglmnet <- function(x, ...) {

  # Find the index of the optimal lambda in the glmnet object
  indexMatch <- order(abs(x$lambda -
                          x$optimalParms["lambda"]))[1]

  # Assemble the optimal parameters, including the df and the % deviance
  op <- matrix(c(x$df[indexMatch],
                 x$dev.ratio[indexMatch],
                 x$optimalParms),
               nrow = 1,
               dimnames = list(NULL, c("Df", "%Dev",
                                       names(x$optimalParms))))

  # Add in the loss estimates if they are present
  if (!is.null(x$lossEstimates)) {

    le <- matrix(c(mean(x$lossEstimates$ExpectedLoss),
                   sd(x$lossEstimates$ExpectedLoss)),
                 nrow = 1,
                 dimnames = list(NULL, c("mean.ExpectedLoss", "sd.ExpectedLoss")))

    op <- cbind(op, le)

  }

  # Print the optimal parms
  cat("The optimal parameter values for the elastic net logistic regression fit: \n")
  print(op)

  # Invisibly return the optimal parms matrix
  invisible(op)

} # print.LRCglmnet

##' @method plot LRCglmnet
##'
##' @describeIn LRCglmnet Produces a pairs plot of the
##' \eqn{(\alpha, \lambda, \tau)} triples and their univariate histograms that
##' were identified as optimal for each of the monte-carlo replicates of the
##' cross-validation partitioning.  This can provide a sense of how stable the estimates
##' are across the cross-validation replicates.  
##'
##' @param \dots Additional arguments to default S3 method \code{\link{pairs}}, used only by the
##' \code{plot} method.  Ignored by the \code{print}, \code{coef}, and \code{predict} methods.
##' 
##' @export

plot.LRCglmnet <- function(x, ...){

  ## put histograms on the diagonal
  panelHistogram <- function(x, ...) {

      usr <- par("usr")
      on.exit(par(usr))
      par(usr = c(usr[1:2], 0, 1.5) )
      h <- hist(x, plot = FALSE)
      breaks <- h$breaks
      nB <- length(breaks)
      y <- h$counts
      y <- y / max(y)
      rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
      
  }

  # Default parameter values
  defaultParms <- list(x = x$parms[,c("alpha", "lambda", "tau")],
                       labels = c(expression(alpha), expression(lambda), expression(tau)),
                       cex = 1.5,
                       pch = 1,
                       cex.labels = 2,
                       font.labels = 2,
                       bg = "light blue",
                       diag.panel = panelHistogram,
                       main = paste("Optimal LRCglmnet parameters for",
                                    NROW(x$parms),
                                    "cross-validation replicates"))


  # Create the parms list
  parmsList <- list(...)

  # Add in default parms not already present in parmslist
  parmsList <- c(parmsList, defaultParms[setdiff(names(defaultParms), names(parmsList))])

  # Make the pairs plot
  do.call(pairs, parmsList)

  invisible(NULL)

} # plot.LRCglmnet


##' @method coef LRCglmnet
##'
##' @describeIn LRCglmnet Calls the \code{\link{predict}} method in \pkg{glmnet}
##' on the fitted glmnet object and returns a named vector of the logistic
##' regression coefficients using the optimal values of \eqn{\alpha} and \eqn{\lambda}.
##'
##' @param object For the \code{coef} and \code{predict} methods:  an object of class
##' \code{LRCglmnet}, returned by \code{LRCglmnet},
##' which contains the optimally-trained elastic net logistic regression classifier
##' 
##' @export

coef.LRCglmnet <- function(object, ...) {

  if (!inherits(object, "glmnet")) {
    stop("Unexpected error.  The 'object' does not inherit from 'glmnet'")
  }

  # Reset the class so that predicting methods work more easily
  class(object) <- setdiff(class(object), "LRCglmnet")

  # Verify the optimal lambda is in there (it should be)
  if (!(object$optimalParms[["lambda"]] %in% object$lambda)) {
    stop("Unexpected error.  The optimal value of lambda was not in 'LRCglmnet_ojbect$lambda'")
  }

  # Get the matrix of coefs for the optimal lambda
  coefs <- as.matrix(predict(object,
                             s = object$optimalParms[["lambda"]],
                             type = "coefficients"))

  # Remove the 0's
  return(coefs[coefs[,1] != 0,])

} # coef.LRCglmnet


##' @importFrom glmnet predict.lognet
##'
##' @method predict LRCglmnet
##' 
##' @describeIn LRCglmnet Predict (or classify) new data using a fitted glmnet logistic regression classifier
##' Returns an object of class \code{LRCpred} (which inherits
##' from \code{data.frame}) that contains the predicted class for each observation.  The columns indicated
##' by \code{truthCol} and \code{keepCols} are included if they were requested.
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
##' @export

predict.LRCglmnet <- function(object,
                              newdata,
                              truthCol = NULL,
                              keepCols = NULL,
                              ...) {

  # Verify it inherits from the lognet class
  if (!inherits(object, "lognet")) {
    stop("Unexpected error:  Object of class 'LRCglmnet' does not inherit from 'lognet'")
  }

  # Switching from column numbers to column names if necessary
  if (!is.null(truthCol) & is.numeric(truthCol)) {
     truthCol <- colnames(newdata)[truthCol]
  }

  if (!is.null(keepCols) & is.numeric(keepCols)) {
     keepCols <- colnames(newdata)[keepCols]
  }

  # Verify the levels of truthCol match the class names in the object
  if (!is.null(truthCol)) {

    # It needs to be a factor
    newdata[,truthCol] <- as.factor(newdata[,truthCol])

    if (!setequal(levels(newdata[,truthCol]), object$classnames))
      warning("The class labels in the 'truthCol' do not match those ",
              "in the 'object'")

  }

  # Get the predictor names expected by the object
  predictorNames <- object$beta@Dimnames[[1]]

  # Make sure all the predictor names are in the newdata
  if (!all(predictorNames %in% colnames(newdata)))
   stop("The following predictors are expected by 'object' but are not\n",
        "present in 'newdata'\n'",
        paste(setdiff(predictorNames, colnames(newdata)), collapse = "', '"), "'\n")

  # Prepare newdata for prediction
  nd <- as.matrix(newdata[,predictorNames])

  if (!is.numeric(nd))
    stop("One or more of the predictor columns in 'newdata' is/are not numeric")

  # Get the original glmnet object
  glmnetObject <- object[-which(names(object) == "optimalParms")]
  class(glmnetObject) <- setdiff(class(object), "LRCglmnet")


  # Get the numeric (probability) predictions using predict methods from glmnet package
  # using the optimal lambda
  preds <- predict(glmnetObject, nd,
                   s = object$optimalParms["lambda"],
                   type = "response")

  # Dichotomize the prediction using the optimal tau
  predLabels <- factor(preds > object$optimalParms["tau"],
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
                                 optimalParms = object$optimalParms,
                                 classNames = glmnetObject$classnames))


  }

  else {
    attributes(output) <- c(attributes(output),
                            list(modelType = "glmnet",
                                 optimalParms = object$optimalParms,
                                 classNames = glmnetObject$classnames))
  }

  return(output)

} # predict.LRCglmnet

