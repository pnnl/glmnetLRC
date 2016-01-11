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
##' as the set of parameters that minimize the loss function defined by \code{lossMat}.
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
##' as n-1 dummary variables in the matrix.
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
##' alpha. A value of alpha = 1 is the lasso penalty, alpha = 0 is the ridge penalty.
##' Refer to \code{\link{glmnet}} for further information.
##'
##' @param tauVec A sequence of tau threshold values for the
##' logistic regression classifier.
##'
##' @param intercept A logical indicating whether an intercept should be included in glmnet model,
##' passed to the \code{intercept} argument of \code{\link{glmnet}}.
##'
##' @param naFilter The maximum proportion of data observations that can be missing
##' for a given predictor
##' (a column in \code{predictors}). If, for a given predictor, the proportion of
##' sample observations which are NA is greater than \code{naFilter}, the predictor is
##' not included in the elastic net model fitting.
##'
##' @param cvFolds The number of cross validation folds.
##\code{cvFolds = NROW(predictors)}
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
##' own thread if \code{cores = 2}.
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
## It also contains the following additional elements:
## TODO \describe{
##'
##'
## }
##'
## values
## of the optimal parameters (averaged over the cross validation replicates), and it
## contains the parameter estimates from the individual cross validation replicates
## as well.
##'
##' @references
##' Amidan BG, Orton DJ, LaMarche BL, Monroe ME, Moore RJ,
##' Venzin AM, Smith RD, Sego LH, Tardiff MF, Payne SH. 2014.
##' Signatures for Mass Spectrometry Data Quality.
##' Journal of Proteome Research. 13(4), 2215-2222.
##' \url{http://pubs.acs.org/doi/abs/10.1021/pr401143e}
##'
##' @examples
##' # Load the VOrbitrap Shewanella QC data
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
##' \dontrun{LRCglmnet_fit <- LRCglmnet(response, predictors, lM, nJobs = max(1, parallel::detectCores() - 1))}
##'
##' # We'll load the precalculated model fit instead
##' \donttest{data(LRCglmnet_fit)}
##' \dontshow{
##' # Here's a call to LRCglment() that will run quickly for testing purposes
##' ncores <- max(1, parallel::detectCores() - 1)
##' LRCglmnet_fit <- LRCglmnet(response, predictors, lM, nJobs = ncores,
##'                            alphaVec = c(1, 0.5), tauVec = c(0.3, 0.5, 0.7),
##'                            cvReps = ncores)
##' }
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

  # Checks inputs
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

  # A wrapper function for calling single_LRCglmnet via parLapply
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

  # Return the optimal parameters to make graphical output
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

      # Execute the training in parallel
      lossEstimates <-
        Smisc::list2df(Smisc::parLapplyW(seedVec, trainWrapper,
                                         alphaVec = finalParmEstimates[["alpha"]],
                                         tauVec = finalParmEstimates[["tau"]],
                                         lambdaVec = lambdaVec,
                                         lambdaVal = finalParmEstimates[["lambda"]],
                                         expr = expression(library(lrc)),
                                         varlist = neededObjects,
                                         njobs = nJobs),
                       row.names = 1:cvReps)


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


    # Include the loss estimates in the final
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
##' These estimates of the loss are obtained by calculating the standard cross-validation estimate
##' of the loss using the glmnet models fit to the trainining folds using the optimized values of
##' \eqn{(\alpha, \lambda, \tau)}.  The same partitions of the data that were used originally to train the
##' LRC are used again, and an estimate of the loss is obtained for each cross validation replicate. The
##' mean and standard deviation of these loss estimates are then shown by this print method. In addition,
##' all of this same information is returned invisibly as a matrix.
##'
##' @param LRCglmnet_object An object of class \code{LRCglmnet}, returned by \code{LRCglmnet},
##' which contains the optimally-trained elastic net logistic regression classifier
##'
##' @export

print.LRCglmnet <- function(LRCglmnet_object) {

  # Find the index of the optimal lambda in the glmnet object
  indexMatch <- order(abs(LRCglmnet_object$lambda -
                          LRCglmnet_object$optimalParms["lambda"]))[1]

  # Assemble the optimal parameters, including the df and the % deviance
  op <- matrix(c(LRCglmnet_object$df[indexMatch],
                 LRCglmnet_object$dev.ratio[indexMatch],
                 LRCglmnet_object$optimalParms),
               nrow = 1,
               dimnames = list(NULL, c("Df", "%Dev",
                                       names(LRCglmnet_object$optimalParms))))

  # Add in the loss estimates if they are present
  if (!is.null(LRCglmnet_object$lossEstimates)) {

    le <- matrix(c(mean(LRCglmnet_object$lossEstimates$ExpectedLoss),
                   sd(LRCglmnet_object$lossEstimates$ExpectedLoss)),
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
##' @param \dots Additional arguments to default S3 method \code{\link{pairs}}.
##' @export

plot.LRCglmnet <- function(LRCglmnet_object, ...){

  ## put histograms on the diagonal
  panelHistogram <- function(x, ...) {

      usr <- par("usr"); on.exit(par(usr))
      par(usr = c(usr[1:2], 0, 1.5) )
      h <- hist(x, plot = FALSE)
      breaks <- h$breaks; nB <- length(breaks)
      y <- h$counts; y <- y/max(y)
      rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
  }

  # Default parameter values
  defaultParms <- list(x = LRCglmnet_object$parms[,c("alpha", "lambda", "tau")],
                       labels = c(expression(alpha), expression(lambda), expression(tau)),
                       cex = 1.5,
                       pch = 1,
                       cex.labels = 2,
                       font.labels = 2,
                       bg = "light blue",
                       diag.panel = panelHistogram,
                       main = paste("Optimal LRCglmnet parameters for",
                                    NROW(LRCglmnet_object$parms),
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
##' @export

coef.LRCglmnet <- function(LRCglmnet_object) {

  if (!inherits(LRCglmnet_object, "glmnet")) {
    stop("Unexpected error.  The 'LRCglmnet_object' does not inherit from 'glmnet'")
  }

  # Reset the class so that predicting methods work more easily
  class(LRCglmnet_object) <- setdiff(class(LRCglmnet_object), "LRCglmnet")

  # Verify the optimal lambda is in there (it should be)
  if (!(LRCglmnet_object$optimalParms[["lambda"]] %in% LRCglmnet_object$lambda)) {
    stop("Unexpected error.  The optimal value of lambda was not in 'LRCglmnet_ojbect$lambda'")
  }

  # Get the matrix of coefs for the optimal lambda
  coefs <- as.matrix(predict(LRCglmnet_object,
                             s = LRCglmnet_object$optimalParms[["lambda"]],
                             type = "coefficients"))

  # Remove the 0's
  return(coefs[coefs[,1] != 0,])

} # coef.LRCglmnet

