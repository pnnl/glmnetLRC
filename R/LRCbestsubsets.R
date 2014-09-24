##' Best subsets logistic regression classifier (LRC) with arbitrary loss function
##'
##' @description
##' This functions uses the \code{\link{bestglm}}
##' function from the \pkg{bestglm}
##' package to fit a best subsets logistic regression model and then estimate an optimal
##' binary classification threshold using cross validation, where optimality is defined
##' by minimizing an arbitrary, user-specified discrete loss function.
##'
##' @details
##' For a given partition of the training data, cross validation is
##' performed to estimate the optimal values of
##' \eqn{\tau}
##' which is used to dichotomize the probability predictions of the
##' logistic regression model into binary outcomes.
##' (Specifically, if the probability an observation
##' belongs to the second level of \code{truthLabels} exceeds \eqn{\tau}, it is
##' classified as belonging to that second level).  In this case, optimality is defined
##' as the the value of \eqn{\tau} that minimize the loss function defined by
##' \code{lossMat}.
##'
##' \code{LRCbestsubsets} searches for the optimal values of \eqn{\tau} by
##' fitting the best subsets logistic regression model and then calculating
##' the expected loss for each \eqn{\tau} in \code{tauVec}, and the
##' value of \eqn{\tau} that results in the lowest risk designates the
##' optimal threshold for a given cross-validation partition of the data.
##'
##' This process is repeated \code{cvReps} times, where each time a different random
##' partition of the data is created using its own seed, resulting in another
##' 'optimal' estimate of \eqn{\tau}.  The final estimate of
##' \eqn{\tau} is the median of those estimates.
##'
##' The final best subsets logistic regression classfier is given by estimating
##' the best subsets regression
##' coefficients using all the training data and then using the optimal \eqn{\tau} to
##' dichotomize the probability predictions from the logistic regression model into
##' one of two binary categories.
##' 
##' @author Landon Sego
##'
##' @rdname LRCbestsubsets
##'
##' @export
##'
##' @param truthLabels A factor with two levels containing the true labels for each
##' observation.
##' If it is more desirable to correctly predict one of the two classes over the other,
##' the second level of this factor should be the class you are most interested in
##' predicting correctly.
##'
##' @param predictors A matrix whose columns are the explanatory regression variables
##'
##' @param lossMat A loss matrix of class \code{lossMat}, produced by
##' \code{\link{lossMatrix}}, that specifies the penalties for classification errors.
##'
##' @param weight The observation weights that are passed to \code{\link{glm}}.
##' The default value is 1 for each observation. Refer to the \code{weights} arguments of
##' \code{\link{glm}} for further information.
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
##' \code{\link{makeCluster}} in package \pkg{parallel}. The \code{cores} argument is
##' ignored when a \code{cluster} is provided.
##'
##' @param verbose A logical, set to \code{TRUE} to receive messages regarding
##' the progress of the training algorithm.
##'
##' @param \dots Additional named arguments to \code{\link{bestglm}} and \code{\link{glm}}.
##'
##' @return
##' Returns an object of class \code{LRCbestsubsets}, which
##' inherits from classes \code{glm} and \code{lm}.  It contains the
##' object returned by \code{\link{glm}} that has been fit to all the data using
##' the the best predictors identified by \code{\link{bestglm}}. It also contains the values
##' of the optimal estimates of \eqn{\tau} from the individual cross validation replicates,
##' along with the median of those estimates, which constitutes the overall estimate of
##' the optimal \eqn{\tau}.
##'
##' Methods \code{print}, \code{summary}, \code{plot}, and \code{coef} are provided.
##' In addition, many of the S3 methods for \code{glm} and \code{lm} can be applied to
##' the object returned by \code{LRCbestsubets}.  See \code{methods(class = "glm")} or
##' \code{methods(class = "lm")} to see a listing of these methods.
##'
## @references
##'
##' @examples
##' # Load the Mojave data
##' data(Mojave)
##' str(Mojave)
##' 
##' # Here we select the predictor variables (remove the location variables and the response)
##' predictors <- Mojave[,-c(1,2,11)]
##' 
##' # Create a vector for the response (presence/absence of cheat grass)
##' cheat <- Mojave$cheatGrass
##' 
##' # "0" is no cheatgrass observed and "1" is cheatgrass.  Note how "1" is the second level
##' # of the factor. This second level is the level for which the summary methods calculate
##' # the sensitivity. So the factor coding here is perfect, since we are most interested
##' # in predicting the presence of cheatgrass with greatest sensitivity (power).
##' levels(cheat)
##' 
##' # Specify the loss matrix. In this example, we specify the penalty for missing
##' # cheatgrass as 2, while the penalty for predicting it falsely is 1.
##' lM <- lossMatrix(c("0","0","1","1"),
##'                  c("0","1","0","1"), 
##'                  c(0,   1,  2,  0))
##' print(lM)
##' 
##' # Train the best subsets logistis regression classifier (in particular, identify
##' # the optimal threshold, tau, that minimizes the loss).  As this takes some time,
##' # we'll skip this step 
##' \dontrun{LRCbestsubsets_fit <- LRCbestsubsets(cheat, predictors, lM,
##'                                               cvReps = 100, cvFolds = 5,
##'                                               cores = max(1, detectCores() - 1))}
##' 
##' 
##' # We'll load the precalculated model fit instead
##' \donttest{data(LRCbestsubsets_fit)}
##' 
##' \dontshow{
##' # Here is a call to LRCbestsubsets() that will run quickly for testing purposes
##' LRCbestsubsets_fit <- LRCglmnet((cheat, predictors, lM, cvReps = 3,
##'                                  cvFolds = 3, cores = max(1, detectCores() - 1),
##'                                  tauVec = c(0.4, 0.5)))
##' }
##' 
##' 
##' # Demonstrate the various methods for LRCbestsubsets() output
##' # (print, summary, plot, coef)
##' print(LRCbestsubsets_fit)
##' 
##' # Assigning the output of print extracts the set of optimal taus that
##' # were identified during the cross validation replicates
##' o <- print(LRCbestsubsets_fit)
##' o
##' 
##' # Sumarize the best fit LRC
##' summary(LRCbestsubsets_fit)
##' 
##' # Plot the loss as a function of the optimal taus for each cross validation
##' # replicate.  Also plot the bestsubsets logistic regression glm object.
##' plot(LRCbestsubsets_fit)
##' 
##' # Display the coefficients
##' coef(LRCbestsubsets_fit)
##' 
##' # Make predictions using final model on the training data
##' out <- predict(LRCbestsubsets_fit, Mojave, truthCol = "cheatGrass")
##' 
##' head(out)
##'
##' # Calculate the performance of predictions
##' summary(out)


## TODO:  Test with factor predictors
## TODO:  What

LRCbestsubsets <- function(truthLabels, predictors, lossMat,
                           weight = rep(1, NROW(predictors)),
                           tauVec = seq(0.1, 0.9, by = 0.05),
                           naFilter = 0.6,
                           cvFolds = 5,
                           cvReps = 100,
                           masterSeed = 1,
                           cores = 1,
                           cluster = NULL,
                           verbose = FALSE,
                           ...) {

  
  # Checks on inputs
  stopifnot(is.factor(truthLabels),
#            is.matrix(predictors), I don't think a matrix is necessary for GLM
#            is.numeric(predictors),
            length(levels(truthLabels)) == 2,
            NCOL(predictors) > 1,
            length(truthLabels) == NROW(predictors),
            inherits(lossMat, "lossMat"),
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
  d <- dataPrep(truthLabels, predictors, weight, naFilter, verbose)

  # number of observations after filtering for NAs
  n <- length(d$truthLabels)

  # Report the number of observations
  if (verbose)
    cat(n, "observations are available for fitting the LRCbestsubsets model\n")

  # Get the name of the truthLabels
  truthLabelName <- deparse(substitute(truthLabels))

  # Create a combined data matrix in the form required by bestglm()
  dm <- cbind(d$predictors, d$truthLabels)

  # If it's not a data frame, change it
  if (is.matrix(dm))
    dm <- as.data.frame(dm)

  # Create column names for data frame
  colnames(dm) <- c(colnames(d$predictors), truthLabelName)

  # Create the weight column
  weight <- d$weight

  ################################################################################
  # Set up the cluster
  ################################################################################
  
  cl <- createCluster(cvReps, masterSeed, cluster, cores)
  
  # Load the glmnetLRC package on the worker nodes
  clusterEvalQ(cl$cluster, require(glmnetLRC))

  # Export the required objects to the##  worker nodes
  clusterExport(cl$cluster,
                c("dm",
                  "tauVec",
                  "n",
                  "cvFolds",
                  "lossMat",
                  "verbose"),
                envir = environment())
  
  # A wrapper function for calling single_LRCbestsubsets via parLapply
  trainWrapper <- function(seed) {

    single_LRCbestsubsets(dm,
                          lossMat,
                          weight,
                          tauVec,
                          cvFolds,
                          seed,
                          n,
                          verbose,
                          ...)

  } # trainWrapper

  # Excecute the training in parallel
  parmEstimates <- list2df(parLapply(cl$cluster, cl$seedVec, trainWrapper),
                           row.names = 1:cvReps)

  # Now stop the cluster
  stopCluster(cl$cluster)

  ################################################################################
  # Create the final model using the median tau value
  ################################################################################

  # Fit the model using the aggregate parameters
  bestsubsetsFinal <- bestglm(dm, weights = weight, family = binomial, ...)$BestModel
                         
  # Return the optimal parameters to make graphical output
  bestsubsetsFinal$tau <- median(parmEstimates$tau)

  # Return the aggregated optimal taus (for each CV replicate)
  bestsubsetsFinal$optimalTaus <- parmEstimates

  # Add in the class names of the response (truth label)
  bestsubsetsFinal$classnames <- levels(truthLabels)

  # Assign the class
  class(bestsubsetsFinal) <- c("LRCbestsubsets", class(bestsubsetsFinal))

  return(bestsubsetsFinal)

} # LRCbestsubsets

##' @method print LRCbestsubsets
##'
##' @describeIn LRCbestsubsets Displays the overall optimized value of
##' \eqn{\tau} and prints (using \code{print.glm}) the best logistic regression model.
##' Invisibly returns the threshold estimate for each cross validation replicate as a
##' data frame.
##'
##' @param LRCbestsubsets_object An object of class \code{LRCbestsubsets},
##' returned by \code{LRCbestsubsets},
##' which contains the best subsets logistic regression model and the optimal threshold,
##' \eqn{\tau}.
##'
##' @export

print.LRCbestsubsets <- function(LRCbestsubsets_object) {

  # Print the optimal parms
  cat("The optimal threshold (tau) for the best subsets logistic regression fit: \n")
  tau <- LRCbestsubsets_object$tau
  pvar(tau)

  # Print the glm object
  stats:::print.glm(LRCbestsubsets_object)

  # Invisibly return the matrix of optimal tau values
  invisible(LRCbestsubsets_object$optimalTaus)

} # print.LRCbestsubsets


##' @method summary LRCbestsubsets
##'
##' @describeIn LRCbestsubsets Displays the overall optimized value of
##' \eqn{\tau} and the summary (using \code{summary.glm}) of the best logistic regression
##' model. 
##'
##' @export

summary.LRCbestsubsets <- function(LRCbestsubsets_object) {

  # Print the optimal parms
  cat("The optimal threshold (tau) for the best subsets logistic regression fit: \n")
  tau <- LRCbestsubsets_object$tau
  pvar(tau)

  return(stats:::summary.glm(LRCbestsubsets_object))

} # print.LRCbestsubsets


##' @method plot LRCbestsubsets
##'
##' @describeIn LRCbestsubsets Produces a scatter plot of \eqn{\tau} vs. the expected
##' loss for each of the monte-carlo replicates of the
##' cross-validation partitioning.  This can provide a sense of how stable the estimates
##' are across the cross-validation replicates. Also plots the \code{glm} object using
##' \code{plot.glm}
##'
##' @param \dots Additional arguments to default S3 method \code{\link{symbols}}.
##' @export

plot.LRCbestsubsets <- function(LRCbestsubsets_object, ...){

  # Prepare for the symbols plot
  d <- LRCbestsubsets_object$optimalTaus
  da <- aggregate(d, by = list(d$tau, d$ExpectedLoss), length)[,1:3]
  colnames(da) <- c("tau", "expectedLoss", "N")

  # Plot the taus and expected loss
  symbols(da$tau, da$expectedLoss, circles = da$N / nrow(d),
          xlab = expression(tau), ylab = "Expected Loss", ...)

  # Plot the glm object
  out <- LRCbestsubsets_object
  class(out) <- setdiff(class(LRCbestsubsets_object), "LRCbestsubsets")
  plot(out)

  invisible(NULL)
  
} # plot.LRCbestsubsets

##' @method coef LRCbestsubsets
##'
##' @describeIn LRCbestsubsets Calls the \code{\link{coef.glm}} method 
##' on the best subsets regression and returns the logistic regression coefficients
##' 
##' @export

coef.LRCbestsubsets <- function(LRCbestsubsets_object) {

  out <- LRCbestsubsets_object
  class(out) <- setdiff(class(LRCbestsubsets_object), "LRCbestsubsets")
  return(coef(out))

} # coef.LRCbestsubsets

