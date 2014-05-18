##' Elastic net logistic regression classifier (LRC) with arbitrary loss function
##'
##' @description
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
##' \code{glmnetLRC} searches for the optimal values of \eqn{\alpha} and \eqn{\tau} by
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
##' @rdname glmnetLRC
##'
##' @export
##'
##' @param truthLabels A factor with two levels containing the true labels for each observation.
##' If it is more desirable to correctly predict one of the two classes over the other,
##' the second level of this factor should be the class you are most interested in
##' predicting correctly.
##' 
##' @param predictors A matrix whose columns are the explanatory regression variables
##' 
##' @param lossMat A loss matrix of class \code{lossMat}, produced by \code{\link{lossMatrix}},
##' that specifies the penalties for
##' classification errors.  
##' 
##' @param weight The observation weights that are passed to \code{\link{glmnet}}.
##' The default value
##' is 1 for each observation. Refer to \code{\link{glmnet}} for further
##' information.
##'
##' @param alphaVec A vector of elastic net mixing parameters. Refer to
##' \code{\link{glmnet}} for further information.
##' 
##' @param tauVec A sequence of tau threshold values for the 
##' logistic regression classifier.
##' 
##' @param naFilter The maximum proportion of data observations that can be missing
##' for a given predictor
##' (a column in \code{predictors}). If, for a given predictor, the proportion of
##' sample observations which are NA is greater than \code{naFilter}, the predictor is
##' not included in the elastic net model fitting.
##' 
##' @param cvFolds The number of cross validation folds. \code{cvFolds = NROW(predictors)}
##' gives leave-one-out cross validation.
##' 
##' @param cvReps The number of cross validation replicates, i.e., the number
##' of times to repeat the cross validation
##' by randomly repartitioning the data into folds.
##' 
##' @param masterSeed The random seed used to generate unique seeds for
##' each cross validation replicate.
##'
##' @param cores The number of cores on the local host
##' to use in parallelizing the training.  Parallelization
##' takes place at the \code{cvReps} level, i.e., if \code{cvReps = 1}, parallelizing would
##' do no good, whereas if \code{cvReps = 2}, each rep would be run separately in its
##' own thread if \code{cores = 2}.
##' Parallelization is executed using \code{\link{parLapply}} from the
##' \pkg{parallel} package.
##'
##' @param cluster An object that inherits from class \code{cluster} that is returned by
##' \code{\link{makeCluster}} in package \pkg{parallel}. The \code{cores} argument is ignored when a
##' \code{cluster} is provided.
##' 
##' @param verbose A logical, set to \code{TRUE} to receive messages regarding
##' the progress of the training algorithm.
##' 
##' @return
##' Returns an object of class \code{glmnetLRC}, which
##' inherits from classes \code{lognet} and \code{glmnet}.  It contains the
##' object returned by \code{\link{glmnet}} that has been fit to all the data using
##' the optimal parameters \eqn{(\alpha, \lambda, \tau)}. It also contains the values
##' of the optimal parameters (averaged over the cross validation replicates), and it
##' contains the parameter estimates from the individual cross validation replicates
##' as well.
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
##' \dontrun{glmnetLRC_fit <- glmnetLRC(response, predictors, lM, cores = max(1, detectCores() - 1))}
##'
##' # We'll load the precalculated model fit instead
##' \donttest{data(glmnetLRC_fit)}
##' \dontshow{
##' # Here is a call to glmnetGLR that will run quickly for testing purposes
##' ncores <- max(1, detectCores() - 1)
##' glmnetLRC_fit <- glmnetLRC(response, predictors, lM, cores = ncores,
##'                            alphaVec = c(1, 0.5), tauVec = c(0.3, 0.5, 0.7),
##'                            cvReps = ncores)
##' }
##' # Show the optimal parameter values
##' print(glmnetLRC_fit)
##'
##' # Show the plot of all the optimal parameter values for each cross validation replicate
##' plot(glmnetLRC_fit)
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
##' # If predictions are made without the an indication of the ground truth,
##' # the summary is simpler:
##' summary(predict(glmnetLRC_fit, testdata))

glmnetLRC <- function(truthLabels, predictors, lossMat,
                      weight = rep(1, NROW(predictors)),
                      alphaVec = seq(0, 1, by = 0.2),
                      tauVec = seq(0.1, 0.9, by = 0.05),
                      naFilter = 0.6,
                      cvFolds = 5,
                      cvReps = 100,
                      masterSeed = 1,
                      cores = 1,
                      cluster = NULL,
                      verbose = FALSE) {

  # Checks on inputs
  stopifnot(NCOL(predictors) > 1,
            is.matrix(predictors),
            is.numeric(predictors),
            length(unique(truthLabels)) == 2,
            length(truthLabels) == NROW(predictors),
            inherits(lossMat, "lossMat"),
            is.numeric(alphaVec),
            all(alphaVec <= 1) & all(alphaVec >= 0),            
            is.numeric(tauVec),
            all(tauVec < 1) & all(tauVec > 0),
            length(naFilter) == 1,
            is.numeric(naFilter), 
            naFilter < 1 & naFilter > 0,
            length(cvFolds) == 1,
            length(cvReps) == 1,
            is.numeric(cvFolds),
            is.numeric(cvReps),
            cvFolds %% 1 == 0,
            cvFolds >= 2,
            cvFolds <= NROW(predictors),
            cvReps %% 1 == 0,
            cvReps > 0,
            is.numeric(masterSeed))

  # Force the evaluation of the weight object immediately--this is IMPORTANT
  # because of R's lazy evaluation
  force(weight)

  ################################################################################
  # Data preparation
  ################################################################################
  
  # Get the data and filter missing values as neccessary
  d <- glmnetLRC:::dataPrep(truthLabels, predictors, weight, naFilter, verbose)

  # number of observations after filtering for NAs
  n <- length(d$truthLabels)

  # Report the number of observations
  if (verbose)
    cat(n, "observations are available for fitting the glmnetLRC model\n")
  
  ################################################################################
  # Set up cluster
  ################################################################################
  
  if (!is.null(cluster)) {
    if (!inherits(cluster, "cluster"))
      stop("'cluster' must inhererit the 'cluster' class")
  }
    
     
  # If the cluster is to be local, simply give the number of cores. If the 
  # cluster is distributed, then a vector of nodeNames and a corresponding
  # vector denoting the number of cpus to use on each node must be supplied
  if (is.null(cluster)) {
    
   nCores <- detectCores()
      
    if (nCores < cores) {

      warning("Number of requested cores exceeds the number available on the host (",
              nCores, ")\n")

    }

    if (cvReps < cores) {
      
      warning("Number of cross validation replicates is less than the number of\n",
              "requested cores.  Setting number of cores to ", cvReps)

      cores <- cvReps
        
    }
      
    cl <- makeCluster(cores)
      
  } else {
      
    # Cluster has been provided
    cl <- cluster
    
  }

  ################################################################################
  # Train via cross validation
  ################################################################################

  # Create the vector of seeds that will be used in the parallel call
  set.seed(masterSeed)
  seedVec <- unique(as.integer(runif(cvReps * 2, min = 1, max = cvReps * 10)))

  # Make sure the length of seedVec is >= cvReps.  If not, add more seeds
  i <- 0
  
  while ((length(seedVec) < cvReps) & (i < 20)) {
    
    seedVec <- unique(c(seedVec,
                        as.integer(runif((cvReps - length(seedVec)) * 10,
                                         min = 1, max = cvReps * 10))))

    i <- i + 1

  }

  # Randomly select a vector of seeds from the unique set
  seedVec <- sample(seedVec, cvReps)
    
  # Load the glmnetLRC package on the worker nodes
  clusterEvalQ(cl, require(glmnetLRC))

  # Export the required objects to the worker nodes
  clusterExport(cl, c("d",
                      "alphaVec",
                      "tauVec",
                      "n",
                      "cvFolds",
                      "lossMat",
                      "verbose"),
                envir = environment())

  # A wrapper function for calling glmnetLRCsingle via parLapply
  trainWrapper <- function(seed) {
    
    glmnetLRCsingle(d$truthLabels,
                    d$predictors,
                    lossMat,
                    d$weight, 
                    alphaVec,
                    tauVec,
                    cvFolds,
                    seed,
                    n,
                    verbose)
    
  } # trainWrapper


  # Excecute the training in parallel
  parmEstimates <- list2df(parLapply(cl, seedVec, trainWrapper), row.names = 1:cvReps)

  # Now stop the cluster
  stopCluster(cl)

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
  lambdaVec <- sort(c(finalParmEstimates["lambda"],
                      seq(finalParmEstimates["lambda"] + sdLambda,
                          max(finalParmEstimates["lambda"] - sdLambda, 1e-04),
                          length = 50)),
                      decreasing = TRUE)
      
  # Fit the model using the aggregate parameters
  glmnetFinal <- glmnet(d$predictors, d$truthLabels, weights = d$weight,
                        family = "binomial", lambda = lambdaVec,
                        alpha = finalParmEstimates["alpha"])

  # Return the optimal parameters to make graphical output
  glmnetFinal$parms <- parmEstimates

  # Return the aggregated optimal parameters (gridVals)
  glmnetFinal$optimalParms <- finalParmEstimates
      
  # Assign the class
  class(glmnetFinal) <- c("glmnetLRC", class(glmnetFinal))

      
  return(glmnetFinal)
  
} # glmnetLRC 

##' @method print glmnetLRC
##' 
##' @describeIn glmnetLRC Displays the overall optimized values of
##' \eqn{(\alpha, \lambda, \tau)}, with the corresponding degrees of freedom and
##' deviance.  Invisibly returns the same information as a matrix.
##' 
##' @param glmnetLRCobject An object of class \code{glmnetLRC}, returned by \code{glmnetLRC},
##' which contains the optimally-trained elastic net logistic regression classifier
##' 
##' @export 

print.glmnetLRC <- function(glmnetLRCobject) {

  # Find the index of the optimal lambda in the glmnet object
  indexMatch <- order(abs(glmnetLRCobject$lambda -
                          glmnetLRCobject$optimalParms["lambda"]))[1]

  # Assemble the optimal parameters, including the df and the % deviance
  op <- matrix(c(glmnetLRCobject$df[indexMatch],
                 glmnetLRCobject$dev.ratio[indexMatch],
                 glmnetLRCobject$optimalParms),
               nrow = 1,
               dimnames = list(NULL, c("Df", "%Dev",
                                       names(glmnetLRCobject$optimalParms))))

  # Print the optimal parms
  cat("The optimal parameter values for the elastic net logistic regression fit: \n")
  print(op)

  # Invisibly return the optimal parms matrix
  invisible(op)
  
} # print.glmnetLRC

##' @method plot glmnetLRC
##'
##' @describeIn glmnetLRC Produces a pairs plot of the
##' \eqn{(\alpha, \lambda, \tau)} triples and their univariate histograms that
##' were identified as optimal for each of the monte-carlo replicates of the
##' cross-validation partitioning.  This can provide a sense of how stable the estimates
##' are across the cross-validation replicates.  
##'
##' @param \dots Additional arguments to default S3 method \code{\link{pairs}}.
##' @export

plot.glmnetLRC <- function(glmnetLRCobject, ...){

  ## put histograms on the diagonal
  panel.hist <- function(x, ...) {

      usr <- par("usr"); on.exit(par(usr))
      par(usr = c(usr[1:2], 0, 1.5) )
      h <- hist(x, plot = FALSE)
      breaks <- h$breaks; nB <- length(breaks)
      y <- h$counts; y <- y/max(y)
      rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
  }


  # Default parameter values
  defaultParms <- list(x = glmnetLRCobject$parms[,c("alpha", "lambda", "tau")],
                       labels = c(expression(alpha), expression(lambda), expression(tau)),
                       cex = 1.5,
                       pch = 1,
                       cex.labels = 2,
                       font.labels = 2,
                       bg = "light blue",
                       diag.panel = panel.hist,
                       main = paste("Optimal glmnetLRC parameters for",
                                    NROW(glmnetLRCobject$parms),
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
##' @describeIn glmnetLRC Calls the \code{\link{coef.glmnet}} method in \pkg{glmnet}
##' on the fitted glmnet object and returns the logistic regression coefficients
##' using the optimal values of \eqn{\alpha} and \eqn{\lambda}.
##' 
##' @export

coef.glmnetLRC <- function(glmnetLRCobject) {

  glmnet:::coef.glmnet(glmnetLRCobject, s = glmnetLRCobject$optimalParms["lambda"])
  
} # coef.glmnetLRC

