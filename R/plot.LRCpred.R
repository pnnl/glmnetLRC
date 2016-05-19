##' Plot the predictions of logistic regression classifier
##'
##' @author Landon Sego
##' 
##' @method plot LRCpred
##'
##' @param x an object of class \code{LRCpred} returned by \code{\link{predict.glmnetLRC}}.
##'
##' @param pch A vector of length 2 indicating the plotting symbols to be used to differentiate the two true classes.  If
##' \code{truthCol} was not specified in the call to \code{\link{predict.glmnetLRC}}, only the first element is used.
##'
##' @param col A vector of length 2 indicating the colors to be used to differentiate the two true classes.  If
##' \code{truthCol} was not specified in the call to \code{\link{predict.glmnetLRC}}, only the first element is used.
##'
##' @param scale A numeric value in (0, 1] that controls scaling of the horizontal axis.  A value of 1 corresponds to the standard,
##' linear scale.  Values closer to 0 expand the axis near 0 and 1 while shrinking the axis in the neighborhood of 0.5.  Values of
##' \code{scale} near 0 are useful if most of the probability predictions are piled up near 0 and 1.
##'
##' @param legendLoc A character string indicating the position of the legend, or a 2-vector of x,y coordinates for the
##' location of the legend. No legend is drawn if \code{legendLoc = NULL}, or if the \code{truthCol} was not specified in the call to
##' \code{\link{predict.glmnetLRC}}.  The value of \code{legendLoc} is passed to \code{\link{legend}}.
##'
##' @param seed Single numeric value used to generate the random jitter of the vertical axis of the plot.
##' 
##' @param \dots Arguments passed to \code{\link{plot.default}}.
##'
##' @return A plot showing the predicted probabilities of the the logisitic regression classifier, with a vertical bar
##' showing the value of the probability threshold, \eqn{\tau}.
##'
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} and \code{\link{glmnetLRC_fit}} for examples.

plot.LRCpred <- function(x, pch = c(1, 2), col = c("Red", "Blue"), scale = 1, legendLoc = "topleft", seed = 1, ...) {

  # Check arguments
  Smisc::stopifnotMsg(if (is.numeric(scale) & (length(scale) == 1)) {
                        (scale > 0) & (scale <= 1)
                      } else FALSE,
                      "'scale' must be a single numeric value in (0, 1]",

                      is.numeric(seed) & (length(seed) == 1)
                      "'seed' must be a single numeric value"
    
  # Get the truth column
  truthCol <- attributes(x)$truthCol

  # Get the number of obs
  n <- nrow(x)
                      
  # Get the jittered y-value
  set.seed(seed)
  y <- Smisc::linearMap(rnorm(n, sd = 0.5))

  # Indicator of the truth class
  if (is.null(truthCol)) {
      
    pchVal <- pch[1]
    colVal <- col[1]
    
  }
  else {

    # Verify we have two values for pch and col
    Smisc::stopifnotMsg(length(pch) == 2,
                        "'pch' must be a vector of length 2 when 'truthCol' was specified in the call to 'predict.glmnetLRC()'",
                        length(col) == 2,
                        "'pch' must be a vector of length 2 when 'truthCol' was specified in the call to 'predict.glmnetLRC()'")
                        
    # Indicator of second truth class
    classInd <- x[,truthCol] == levels(x[,truthCol])[2]

    # Set the pch
    pchVal <- rep(pch[1], n)
    pchVal[classInd] <- pch[2]

    # Set the col
    colVal <- rep(col[1], n)
    colVal[classInd] <- col[2]

  }
                      
  # Specify the transform for the x-axis
  tr <- function(x) pbeta(x, scale, scale)
  
  # Set up the default plotting parameters
  defaultArgs <- list(x = tr(x$Prob),
                      y = y,
                      pch = pchVal,
                      col = colVal,
                      ylim = range(y),
                      xlim = range(tr(x$Prob)))

  suppliedArgs <- list(...)
                      
                      
} # plot.LRCpred
