##' Construct a lasso or elastic-net logistic regression classifier (LRC) with an arbitrary loss function
##'
##' This function extends the \code{\link{glmnet}} and \code{\link{cv.glmnet}}
##' functions from the \href{http://cran.r-project.org/package=glmnet}{glmnet}
##' package. It uses cross validation to identify optimal elastic-net parameters and a
##' threshold parameter for binary classification, where optimality is defined
##' by minimizing an arbitrary, user-specified discrete loss function.
##'
##' @details
##' For a given partition of the training data, cross validation is
##' performed to estimate the optimal values of
##' \eqn{\alpha} (the mixing parameter of the ridge and lasso penalties) and \eqn{\lambda}
##' (the regularization parameter), as well as the optimal threshold, \eqn{\tau},
##' which is used to dichotomize the probability predictions of the elastic-net
##' logistic regression model into binary outcomes.
##' (Specifically, if the probability an observation
##' belongs to the second level of \code{truthLabels} exceeds \eqn{\tau}, it is
##' classified as belonging to that second level).  In this case, optimality is defined
##' as the set of parameters that minimize the risk, or expected loss, where the
##' loss function created using \code{\link{lossMatrix}}.  The expected loss is calculated such
##' that each observation in the data receives equal weight
##'
##' \code{glmnetLRC()} searches for the optimal values of \eqn{\alpha} and \eqn{\tau} by
##' fitting the elastic-net model at the points of the two-dimensional grid defined by
##' \code{alphaVec} and \code{tauVec}.  For each value of \eqn{\alpha}, the vector of
##' \eqn{\lambda} values is selected automatically by \code{\link{glmnet}} according to its default
##' arguments. The expected loss is calculated for each \eqn{(\alpha,\lambda,\tau)} triple, and the
##' triple giving rise to the lowest risk designates the optimal model for a given
##' cross validation partition, or cross validation replicate, of the data.
##'
##' This process is repeated \code{cvReps} times, where each time a different random
##' partition of the data is created using its own seed, resulting in another
##' 'optimal' estimate of \eqn{(\alpha,\lambda,\tau)}.  The final estimate of
##' \eqn{(\alpha,\lambda,\tau)} is given by the respective medians of those estimates.
##' The final elastic-net logistic regression classfier is given by fitting the regression
##' coefficients to all the training data using the optimal \eqn{(\alpha,\lambda,\tau)}.
##'
##' The methodology is discussed in detail in the online
##' \href{http://pnnl.github.io/docs-glmnetLRC/index.html#mathematical-details}{package documentation}.
##'
##' @author Landon Sego, Alex Venzin
##'
##' @rdname glmnetLRC
##'
##' @export
##' @param truthLabels A factor with two levels containing the true labels for each
##' observation. If it is more desirable to correctly predict one of the two classes over the other,
##' the second level of this factor should be the class you are most interested in
##' predicting correctly.
##'
##' @param predictors A matrix whose columns are the explanatory regression variables.  Note:
##' factors are not currently supported.  To include a factor variable with \emph{n} levels, it must be represented
##' as \emph{n-1} dummy variables in the matrix.
##'
##' @param lossMat Either the character string \code{"0-1"}, indicating 0-1 loss, or a loss
##' matrix of class \code{lossMat}, produced by \code{\link{lossMatrix}}, that specifies
##' the penalties for classification errors.
##'
##' @param lossWeight A vector of non-negative weights used to calculate the expected loss. The default value is 1 for
##' each observation.
##'
##' @param alphaVec A sequence in [0, 1] designating possible values for the elastic-net mixing parameter,
##' \eqn{\alpha}. A value of \eqn{\alpha = 1} is the lasso penalty, \eqn{\alpha = 0} is the ridge penalty.
##' Refer to \code{\link{glmnet}} for further information.
##'
##' @param tauVec A sequence of \eqn{\tau} threshold values in (0, 1) for the
##' logistic regression classifier. For a new observation, if the predicted probability
##' that the observation belongs to the second level
##' of \code{truthLabels} exceeds tau, the observation is classified as belonging
##' to the second level.
##'
##' @param cvFolds The number of cross validation folds.
##' \code{cvFolds = length(truthLabels)} gives leave-one-out (L.O.O.) cross validation,
##' in which case \code{cvReps} is set to \code{1} and \code{stratify} is set to \code{FALSE}.
##'
##' @param cvReps The number of cross validation replicates, i.e., the number
##' of times to repeat the cross validation
##' by randomly repartitioning the data into folds and estimating the tuning parameters.
##' For L.O.O. cross validation, this argument is set to \code{1} as there can only be one
##' possible partition of the data.
##'
##' @param stratify A logical indicating whether stratified sampling should be used
##' to ensure that observations from
##' both levels of \code{truthLabels} are proportionally present in the cross validation
##' folds. In other words, stratification attempts to ensure there are sufficient observations
##' of each level of \code{truthLabels} in each training set to fit the model.
##' Stratification may be required for small or imbalanced data sets.  Note that stratification
##' is not performed for L.O.O (when \code{cvFolds = length(truthLabels)}).
##'
##' @param masterSeed The random seed used to generate unique (and repeatable) seeds for
##' each cross validation replicate.
##'
##' @param nJobs The number of cores on the local host
##' to use in parallelizing the training.  Parallelization
##' takes place at the \code{cvReps} level, i.e., if \code{cvReps = 1}, parallelizing would
##' do no good, whereas if \code{cvReps = 2}, each cross validation replicate would be run
##' separately in its own thread if \code{nJobs = 2}.
##' Parallelization is executed using \href{http://pnnl.github.io/docs-Smisc/rd.html#parlapplyw}{parLapplyW()}
##' from the \href{http://pnnl.github.io/docs-Smisc}{Smisc} package.
##'
##' @param estimateLoss A logical, set to \code{TRUE} to calculate the average loss estimated via
##' cross validation using the optimized parameters \eqn{(\alpha, \lambda, \tau)} to fit the elastic
##' net model for each cross validation fold. This can be computationally expensive,
##' as it requires another cross validation pass through the same partitions of the data, but using only
##' the optimal parameters to estimate the loss for each cross validation replicate.
##'
##' @param verbose For \code{glmetLRC}, a logical to turn on (or off) messages regarding
##' the progress of the training algorithm.  For the \code{print} method, if set to \code{FALSE}, it
##' will suppress printing information about the \code{glmnetLRC} object and only invisibly return
##' the results.
##'
##' @param \dots For \code{glmnetLRC()}, these are additional arguments to \code{\link{glmnet}} in the \code{glmnet} package.
##' Certain arguments of \code{\link{glmnet}} are reserved by the \code{glmnetLRC} package and an error message will make that
##' clear if they are used.  In particular, arguments that control the behavior of \eqn{\alpha} and \eqn{\lambda} are reserved.
##' For the \code{plot} method, the "\dots" are additional arguments to the default S3 method \code{\link{pairs}}.  And for
##' the \code{print}, \code{coef}, \code{predict}, \code{missingpreds}, and \code{extract} methods, the "\dots" are ignored.
##'
##' @return
##' An object of class \code{glmnetLRC}, which
##' inherits from classes \code{lognet} and \code{glmnet}.  It contains the
##' object returned by \code{\link{glmnet}} that has been fit to all the data using
##' the optimal parameters \eqn{(\alpha, \lambda, \tau)}.
##' It also contains the following additional elements:
##' \describe{
##' \item{lossMat}{The loss matrix used as the criteria for selecting optimal tuning parameters}
##' \item{parms}{A data fame that contains the tuning parameter estimates for \eqn{(\alpha, \lambda, \tau)} that minimize
##' the expected loss for each cross validation replicate.  Used by the \code{plot} method.}
##' \item{optimalParms}{A named vector that contains the final estimates of \eqn{(\alpha, \lambda, \tau)}, calculated as the
##' element-wise median of \code{parms}}
##' \item{lossEstimates}{If \code{estimateLoss = TRUE}, this element is a data frame with the expected loss
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
##' Friedman J, Hastie T, Tibshirani R. 2010. Regularization Paths for Generalized
##' Linear Models via Coordinate Descent. Journal of Statistical Software.
##' 33(1), 1-22.
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
##' # Train the elastic-net classifier (we don't run it here because it takes a long time)
##' \dontrun{
##' glmnetLRC_fit <- glmnetLRC(response, predictors, lossMat = lM, estimateLoss = TRUE,
##'                            nJobs = parallel::detectCores())
##' }
##'
##' # We'll load the precalculated model fit instead
##' data(glmnetLRC_fit)
##'
##' # Show the optimal parameter values
##' print(glmnetLRC_fit)
##'
##' # Show the coefficients of the optimal model
##' coef(glmnetLRC_fit)
##'
##' # Show the plot of all the optimal parameter values for each cross validation replicate
##' plot(glmnetLRC_fit)
##'
##' # Extract the 'glmnet' object from the glmnetLRC fit
##' glmnetObject <- extract(glmnetLRC_fit)
##'
##' # See how the glmnet methods operate on the object
##' plot(glmnetObject)
##'
##' # Look at the coefficients for the optimal lambda
##' coef(glmnetObject, s = glmnetLRC_fit$optimalParms["lambda"] )
##'
##' # Load the new observations
##' data(testdata)
##'
##' # Use the trained model to make predictions about
##' # new observations for the response variable.
##' new <- predict(glmnetLRC_fit, testdata, truthCol = "Curated_Quality", keepCols = 1:2)
##' head(new)
##'
##' # Now summarize the performance of the model
##' summary(new)
##'
##' # And plot the probability predictions of the model
##' plot(new, scale = 0.5, legendArgs = list(x = "topright"))
##'
##' # If predictions are made without an indication of the ground truth,
##' # the summary is necessarily simpler:
##' summary(predict(glmnetLRC_fit, testdata))

glmnetLRC <- function(truthLabels, predictors,
                      lossMat = "0-1",
                      lossWeight = rep(1, NROW(predictors)),
                      alphaVec = seq(0, 1, by = 0.2),
                      tauVec = seq(0.1, 0.9, by = 0.05),
                      cvFolds = 5,
                      cvReps = 100,
                      stratify = FALSE,
                      masterSeed = 1,
                      nJobs = 1,
                      estimateLoss = FALSE,
                      verbose = FALSE,
                      ...) {

  # Force the evaluation of the weight object immediately--this is IMPORTANT
  # because of R's lazy evaluation
  force(lossWeight)

  # Check argument types
  Smisc::stopifnotMsg(
    is.factor(truthLabels), "'truthLabels' must be a factor",
    is.matrix(predictors), "'predictors' must be a matrix",
    is.numeric(predictors), "'predictors' must be a numeric",
    is.numeric(lossWeight), "'lossWeight' must be a numeric vector",
    is.numeric(alphaVec), "'alphaVec' must be numeric",
    is.numeric(tauVec), "'tauVec' must be numeric",
    is.numeric(cvFolds), "'cvFolds' must be numeric",
    is.numeric(cvReps), "'cvReps' must be numeric",
    is.logical(stratify), "'stratify' must be TRUE or FALSE",
    is.numeric(masterSeed), "'masterSeed' must be numeric",
    is.numeric(nJobs), "'nJobs' must be numeric",
    is.logical(estimateLoss), "'estimateLoss' must be TRUE or FALSE",
    is.logical(verbose), "'verbose' must have length 1")

  # Further checks of argument values
  Smisc::stopifnotMsg(
    length(levels(truthLabels)) == 2, "'truthLabels' must have 2 levels",
    all(complete.cases(truthLabels)), "'truthLabels' cannot contain missing values",
    NCOL(predictors) > 1, "'predictors' must have at least 2 columns",
    all(complete.cases(predictors)),  "'predictors' cannot contain missing values",
    length(truthLabels) == NROW(predictors), "the length of 'truthLabels' must match the number of rows in 'predictors'",
    length(lossWeight) == NROW(predictors), "the length of 'lossWeight' must match the number of rows in 'predictors'",
    all(lossWeight >= 0), "All values of 'lossWeight' must be non-negative",
    !all(lossWeight == 0), "Not all of the 'lossWeight' values can be zero",
    all(alphaVec <= 1) & all(alphaVec >= 0), "All values of 'alphaVec' must be in [0, 1]",
    all(tauVec < 1) & all(tauVec > 0), "All values of 'alphaVec' must be in (0, 1)",
    length(cvFolds) == 1, "'cvFolds' must be of length 1",
    cvFolds %% 1 == 0, "'cvFolds' must be an integer",
    cvFolds >= 2, "'cvFolds' must be 2 or greater",
    cvFolds <= NROW(predictors), "'cvFolds' cannot be larger than the number of rows in 'predictors'",
    length(cvReps) == 1, "'cvReps' must be of length 1",
    cvReps %% 1 == 0, "'cvReps' must be an integer",
    cvReps > 0, "'cvReps' must be 1 or larger",
    length(stratify) == 1, "'stratify' must be of length 1",
    length(masterSeed) == 1, "'masterSeed' must be of length 1",
    nJobs >= 1, "'nJobs' must be 1 or greater",
    nJobs %% 1 == 0, "'nJobs' must be an integer",
    length(estimateLoss) == 1, "'estimateLoss' must have length of 1",
    length(verbose) == 1, "'verbose' must be TRUE or FALSE")

  # Check the loss matrix
  lmGood <- FALSE

  if (is.character(lossMat)) {
    if (length(lossMat) == 1) {
      if (lossMat == "0-1") {

        # Create the 0-1 loss matrix
        lossMat <- lossMatrix(rep(levels(truthLabels), each = 2),
                              rep(levels(truthLabels), 2),
                              c(0, 1, 1, 0))
        lmGood <- TRUE

      }
    }
  }
  else if (inherits(lossMat, "lossMat")) {
    lmGood <- TRUE
  }
  if (!lmGood) {
    stop("'lossMat' must be either '0-1' or an object of class 'lossMat' returned by 'lossMatrix()'")
  }


  # Gather arguments for glmnet
  glmnetArgs <- list(...)

  # Check the glmnet arguments
  if (length(glmnetArgs)) {

    # Get the names of the args provided by the user
    userArgs <- names(glmnetArgs)

    # Make sure the user hasn't provided any of the glmnet args that are 'off-limits'
    offLimits <- c("x", "y", "family", "alpha", "nlambda", "lambda.min.ratio", "lambda",
                   "type.gaussian", "type.multinomial", "standardize.response")

    # Verify none of the off limit args have been used
    if (any(userArgs %in% offLimits)) {
      stop("The following arguments to glmnet::glmnet() are controlled (or not relevant) and should not be supplied to '...':\n",
           "'", paste(offLimits, collapse = "', '"), "'\n")
    }

    # Get the name of the glmnet args
    defaultArgs <- names(formals(glmnet::glmnet))

    # Verify that all the glmnent args provided match glmnet
    if (!all(userArgs %in% defaultArgs)) {
      stop("The following do not match the arguments in glmnet::glmnet():\n",
           "'", paste(setdiff(userArgs, defaultArgs), collapse = "', '"), "'\n")
    }

  }

  # Create the full list of glmnetArgs that doesn't involve the lambda or alpha parameters
  glmnetArgs <- c(list(x = predictors, y = truthLabels, family = "binomial"), glmnetArgs)
  class(glmnetArgs) <- c("glmnetArgs", class(glmnetArgs))

  ################################################################################
  # Data preparation
  ################################################################################

  # number of observations after filtering for NAs
  n <- length(truthLabels)

  # Report the number of observations
  if (verbose) {
    cat(n, "observations are available for fitting the glmnetLRC model\n")
  }


  # A wrapper function for calling single_glmnetLRC() via Smisc::parLapplyW()
  trainWrapper <- function(testFolds,
                           alphaVec = 1,
                           tauVec = 0.5,
                           lambdaVal = NULL,
                           lambdaVec = NULL) {

    single_glmnetLRC(glmnetArgs,
                     lossMat,
                     lossWeight,
                     alphaVec,
                     tauVec,
                     cvFolds,
                     testFolds,
                     n,
                     verbose,
                     lambdaVal = lambdaVal,
                     lambdaVec = lambdaVec)

  } # trainWrapper

  # Generate the test sets that will be passed into the cv routines
  testSets <- generateTestSets(truthLabels, cvFolds, cvReps, masterSeed, stratify)

  # nJobs should not be larger than cvReps
  nJobs <- min(nJobs, cvReps)

  ################################################################################
  # Run in parallel
  ################################################################################
  if (nJobs > 1) {

    # Names of objects that will be needed in the cluster
    neededObjects <- c("glmnetArgs",
                       "lossMat",
                       "lossWeight",
                       "cvFolds",
                       "n",
                       "verbose")

    # Execute the training in parallel
    pe <- Smisc::parLapplyW(testSets, trainWrapper,
                            alphaVec = alphaVec,
                            tauVec = tauVec,
                            njobs = nJobs,
                            expr = expression(library(glmnetLRC)),
                            varlist = neededObjects)

    # Collapse results to a data frame
    parmEstimates <- Smisc::list2df(pe, row.names = 1:cvReps)

  }

  ################################################################################
  # Single thread
  ################################################################################

  else {

    parmEstimates <- Smisc::list2df(lapply(testSets, trainWrapper,
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
  # regression model. Per the documentation in glmnetfit, it does better
  # with a sequence of lambdas during the fitting.  But the predict method only
  # operates on the optimal lambda
  sdLambda <- sd(parmEstimates$lambda)
  lambdaVec <- sort(c(finalParmEstimates[["lambda"]],
                      seq(finalParmEstimates[["lambda"]] + sdLambda,
                          max(finalParmEstimates[["lambda"]] - sdLambda, 1e-04),
                          length = 50)),
                      decreasing = TRUE)

  # Fit the model using the aggregate parameters
  glmnetFinal <- do.call(glmnet::glmnet,
                         c(glmnetArgs, list(lambda = lambdaVec, alpha = finalParmEstimates[["alpha"]])))

  # Add the loss matrix
  glmnetFinal$lossMat <- lossMat

  # Return the optimal parameters for graphical output
  glmnetFinal$parms <- parmEstimates

  # Return the finalized tuning parameters (median of the cross validation replicates)
  glmnetFinal$optimalParms <- finalParmEstimates

  ################################################################################
  # If we are to estimate the loss by averaging the cross validation loss over
  # all the CV replicates, using the optimal parameters for each fit to a CV
  # training set.  Will use the same random seeds as before to construct
  # the training/testing sets.
  ################################################################################

  if (estimateLoss) {

    ################################################################################
    # Run on the cluster which was set up earlier
    ################################################################################
    if (nJobs > 1) {

      # Execute the loss calculation in parallel
      le <- Smisc::parLapplyW(testSets, trainWrapper,
                              alphaVec = finalParmEstimates[["alpha"]],
                              tauVec = finalParmEstimates[["tau"]],
                              lambdaVec = lambdaVec,
                              lambdaVal = finalParmEstimates[["lambda"]],
                              expr = expression(library(glmnetLRC)),
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
        Smisc::list2df(lapply(testSets, trainWrapper,
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
  class(glmnetFinal) <- c("glmnetLRC", class(glmnetFinal))


  return(glmnetFinal)

} # glmnetLRC

##' @method print glmnetLRC
##'
##' @describeIn glmnetLRC Displays the overall optimized values of
##' \eqn{(\alpha, \lambda, \tau)}, with the corresponding degrees of freedom and
##' deviance for the model fit to all the data using the optimzed parameters.  If \code{estimateLoss = TRUE}
##' when \code{glmnetLRC()} was called, the mean and standard deviation of the expected loss are also shown.
##' In addition, all of this same information is returned invisibly as a matrix. Display of the information
##' can be suppressed by setting \code{verbose = FALSE} in the call to \code{print}.
##'
##' @param x For the \code{print} and \code{plot} methods:  an object of class \code{glmnetLRC} (returned
##' by \code{glmnetLRC()}), which contains the optimally-trained elastic-net logistic regression classifier.
##'
##' @export

print.glmnetLRC <- function(x, verbose = TRUE, ...) {

  # Check verbose
  Smisc::stopifnotMsg(is.logical(verbose) & (length(verbose) == 1),
                      "'verbose' must be TRUE or FALSE")
    
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
  if (verbose) {
    cat("The optimal parameter values for the elastic-net logistic regression fit: \n")
    print(op)
  }

  # Invisibly return the optimal parms matrix
  invisible(op)

} # print.glmnetLRC

##' @method plot glmnetLRC
##'
##' @describeIn glmnetLRC Produces a pairs plot of the tuning parameters
##' \eqn{(\alpha, \lambda, \tau)} and their univariate histograms that
##' were identified as optimal for each of of the cross validation replicates.
##' This can provide a sense of the stability of the estimates of the tuning
##' parameters.
##'
##' @export

plot.glmnetLRC <- function(x, ...){

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
                       main = paste("Optimal glmnetLRC parameters for",
                                    NROW(x$parms),
                                    "cross validation replicates"))


  # Create the parms list
  parmsList <- list(...)

  # Add in default parms not already present in parmslist
  parmsList <- c(parmsList, defaultParms[setdiff(names(defaultParms), names(parmsList))])

  # Make the pairs plot
  do.call(pairs, parmsList)

  invisible(NULL)

} # plot.glmnetLRC


##' @method coef glmnetLRC
##'
##' @describeIn glmnetLRC Calls the \code{predict} method in \code{glmnet}
##' on the fitted glmnet object and returns a named vector of the non-zero elastic-net logistic
##' regression coefficients using the optimal values of \eqn{\alpha} and \eqn{\lambda}.
##'
##' @param object For the \code{coef}, \code{predict}, and \code{extract} methods:
##' an object of class \code{glmnetLRC} (returned by \code{glmnetLRC()})
##' which contains the optimally-trained elastic-net logistic regression classifier.
##'
##' @param tol A small positive number, such that coefficients with an absolute value smaller than
##' \code{tol} are not returned.
##'
##' @export

coef.glmnetLRC <- function(object, tol = 1e-10, ...) {

  if (!inherits(object, "glmnet")) {
    stop("Unexpected error.  The 'object' does not inherit from 'glmnet'")
  }

  # Verify tol
  Smisc::stopifnotMsg(if (is.numeric(tol)) {
                        if (length(tol) == 1) {
                          tol > 0
                        } else FALSE
                      } else FALSE,
                      "'tol' should be a small, positive number")
  
  # Verify the optimal lambda is in there (it should be)
  if (!(object$optimalParms[["lambda"]] %in% object$lambda)) {
    stop("Unexpected error.  The optimal value of lambda was not in 'glmnetLRC_ojbect$lambda'")
  }

  # Get the matrix of coefs for the optimal lambda
  coefs <- as.matrix(predict(extract(object),
                             s = object$optimalParms[["lambda"]],
                             type = "coefficients"))

  # Remove the 0's
  zeroCoefs <- abs(coefs[,1]) < tol
  
  return(coefs[!zeroCoefs,])

} # coef.glmnetLRC


##' @importFrom glmnet predict.lognet
##'
##' @method predict glmnetLRC
##'
##' @describeIn glmnetLRC Predict (or classify) new data from an \code{glmnetLRC} object.
##' Returns an object of class \code{LRCpred} (which inherits
##' from \code{data.frame}) that contains the predicted probabilities (\code{Prob}) and class (\code{predictClass})
##' for each observation.  The \code{Prob} column corresponds to the predicted probability that an observation belongs
##' to the second level of \code{truthLabels}. The columns indicated by \code{truthCol} and \code{keepCols} are included
##' if they were requested.  The \code{LRCpred} class has two methods:  \code{\link{summary.LRCpred}} and \code{\link{plot.LRCpred}}.
##'
##' @param newdata A dataframe or matrix containing the new set of observations to
##' be predicted, as well as an optional column of true labels.
##' \code{newdata} should contain all of the column names that were used
##' to fit the elastic-net logistic regression classifier.
##'
##' @param truthCol The column number or column name in \code{newdata} that contains the
##' true labels, which should be a factor (and this implies \code{newdata} should be a dataframe if \code{truthCol} is provided).
##' Optional.
##'
##' @param keepCols A numeric vector of column numbers (or a character vector of
##' column names) in \code{newdata} that will be 'kept' and returned with the predictions. Optional.
##'
##' @export

predict.glmnetLRC <- function(object,
                              newdata,
                              truthCol = NULL,
                              keepCols = NULL,
                              ...) {

  # Verify it inherits from the lognet class
  if (!inherits(object, "lognet")) {
    stop("Unexpected error:  Object of class 'glmnetLRC' does not inherit from 'lognet'")
  }

  # Check newdata
  Smisc::stopifnotMsg(is.matrix(newdata) | is.data.frame(newdata),
                      "'newdata' must be a matrix or dataframe")
  
  # Switching from column numbers to column names if necessary
  truthCol <- Smisc::selectElements(truthCol, colnames(newdata))
  keepCols <- Smisc::selectElements(keepCols, colnames(newdata))

  # Verify the levels of truthCol match the class names in the object
  if (!is.null(truthCol)) {

    # It needs to be a factor
    if (!is.factor(newdata[,truthCol])) {
      stop("'truthCol', if provided, needs to be a factor, and, consequently, 'newdata' should be a data frame")
    }

    if (!setequal(levels(newdata[,truthCol]), object$classnames)) {
      warning("The class labels in the 'truthCol' do not match those ",
              "in the 'object'")
    }

  }

  # Make sure all the predictor names are in the newdata
  if (nm <- length(missingpreds(object, newdata))) {

   # Produce the error message
   stop(if (nm > 1) paste("There are", nm, "predictors that are required by 'object' but",
                          "are not present in 'newdata'.\n")
        else paste("There is 1 predictor required by 'object' that is not present in 'newdata'.\n"),
        "  Use 'missingpreds(object, newdata)' to identify ",
        if (nm > 1) "them" else "it")
  }

  # Prepare newdata for prediction
  nd <- as.matrix(newdata[, object$beta@Dimnames[[1]]])

  if (!is.numeric(nd)) {
    stop("One or more of the predictor columns in 'newdata' is/are not numeric")
  }

  # Get the original glmnet object
  glmnetObject <- extract(object)

  # Get the numeric (probability) predictions using predict methods from glmnet package
  # using the optimal lambda
  preds <- predict(glmnetObject, nd,
                   s = object$optimalParms["lambda"],
                   type = "response")

  # Dichotomize the prediction using the optimal tau
  predLabels <- factor(preds > object$optimalParms["tau"],
                       levels = c(FALSE, TRUE),
                       labels = glmnetObject$classnames)

  # If there are rownames in newdata, add them in
  if (!is.null(rn <- rownames(newdata))) {
    names(predLabels) <- rn
  }

  # Combine data
  selCols <- c(truthCol, keepCols)
  if (is.null(selCols)) {
    selCols <- 0
  }
  output <- cbind(preds, predLabels, as.data.frame(Smisc::select(newdata, selCols)))
  colnames(output)[1:2] <- c("Prob", "PredictClass")

  # Assign the class and attributes
  class(output) <- c("LRCpred", class(output))

  attributes(output) <- c(attributes(output),
                          list(truthCol = truthCol,
                               optimalParms = object$optimalParms,
                               classNames = glmnetObject$classnames))

  return(output)

} # predict.glmnetLRC

##' @method missingpreds glmnetLRC
##'
##' @describeIn glmnetLRC Identify the set of predictors in a \code{glmnetLRC} object that are not
##' present in \code{newdata}. Returns a character vector of the missing predictor names. If no predictors are missing,
##' it returns \code{character(0)}.
##'
##' @export

missingpreds.glmnetLRC <- function(object, newdata, ...) {

  Smisc::stopifnotMsg(is.matrix(newdata) | is.data.frame(newdata),
                      "'newdata' must be a matrix or dataframe")
    
  # Get the predictor names expected by the object
  predictorNames <- object$beta@Dimnames[[1]]

  return(setdiff(predictorNames, colnames(newdata)))

} # missingpreds.glmnetLRC


##' @method extract glmnetLRC
##'
##' @describeIn glmnetLRC Extracts the \code{glmnet} object that was fit using the optimal parameter estimates of
##' \eqn{(\alpha, \lambda)}.  Returns an object of class \code{"lognet" "glmnet"} that can be passed to various
##' methods available in the \code{glmnet} package.
##'
##' @export

extract.glmnetLRC <- function(object, ...) {

  # Remove the lossMat, parms, optimalParms, and lossEstimates elements of the object.
  out <- object[-which(names(object) %in% c("lossMat", "parms", "optimalParms", "lossEstimates"))]

  # Remove it's LRCglmnet class.
  class(out) <- class(object)[-which(class(object) == "glmnetLRC")]

  return(out)

} # extract.glmnetLRC

