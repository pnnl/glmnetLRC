##' Plot the predictions of logistic regression classifier
##'
##' @author Landon Sego
##' 
##' @method plot LRCpred
##'
##' @param x an object of class \code{LRCpred} returned by \code{\link{predict.glmnetLRC}}.
##'
##' @param pch A vector of at most length 2 indicating the plotting symbols to be used to differentiate the two true classes.  If
##' \code{truthCol} was not specified in the call to \code{\link{predict.glmnetLRC}}, only the first element is used. This is passed to
##' \code{\link{plot}}.
##'
##' @param col A vector of at most length 2 indicating the colors of the plotted points in order to differentiate the two true classes.  If
##' \code{truthCol} was not specified in the call to \code{\link{predict.glmnetLRC}}, only the first element is used. This is passed to
##' \code{\link{plot}}.
##'
##' @param scale A numeric value in (0, 1] that controls scaling of the horizontal axis.  A value of 1 corresponds to the standard,
##' linear scale.  Values closer to 0 symetrically 'zoom-in' the axis near 0 and 1 while 'zooming-out' the axis in the neighborhood of 0.5.
##' Values of \code{scale} closer to 0 are useful if most of the probability predictions are piled up near 0 and 1.
##'
##' @param seed Single numeric value that sets the seed for the random jitter of the vertical axis of the plot.
##'
##' @param parArgs If desired, a list of named arguments that will be passed to \code{\link{par}} which is called prior to making the plot.
##'
##' @param legendArgs If desired, a list of named arguments that will be passed to \code{\link{legend}}. If
##' \code{truthCol} was not specified in the call to \code{\link{predict.glmnetLRC}}, no legend is drawn.
##'
##' @param lineArgs If desired, a list of named arguments that will be passed to \code{\link{abline}} governing the vertical line that
##' indicates the value of \eqn{\tau}.
##'
##' @param textArgs If desired, a list of named arguments that will be passed to \code{\link{text}} governing the text indicating the
##' value of \eqn{\tau}.
##' 
##' @param \dots Arguments passed to \code{\link{plot.default}}.
##'
##' @return A plot showing the predicted probabilities of the logisitic regression classifier, with a vertical bar
##' showing the value of the probability threshold, \eqn{\tau}.
##'
##' @export
##'
##' @seealso See \code{\link{glmnetLRC}} for an example.

plot.LRCpred <- function(x, pch = c(1, 2), col = c("Blue", "Red"), scale = 1, seed = 1,
                         parArgs = NULL, legendArgs = NULL, lineArgs = NULL, textArgs = NULL, ...) {

  # Check arguments
  Smisc::stopifnotMsg(# pch
                      (length(pch) > 0) & (length(pch) <= 2),
                      "'pch' must be of length 1 or 2",

                      # col
                      (length(col) > 0) & (length(col) <= 2),
                      "'col' must be of length 1 or 2",

                      # scale
                      if (is.numeric(scale) & (length(scale) == 1)) {
                        (scale > 0) & (scale <= 1)
                      } else FALSE,
                      "'scale' must be a single numeric value in (0, 1]",

                      # seed
                      is.numeric(seed) & (length(seed) == 1),
                      "'seed' must be a single numeric value",

                      # parArgs
                      if (!is.null(parArgs)) {
                        if (is.list(parArgs)) {
                          all(names(parArgs) %in% listParNames())
                        } else FALSE
                      } else TRUE,
                      "'parArgs' must be NULL or a list whose names match the arguments of 'par()'",

                      # legendArgs
                      if (!is.null(legendArgs)) {
                        if (is.list(legendArgs)) {
                          all(names(legendArgs) %in% names(formals(legend)))
                        } else FALSE
                      } else TRUE,
                      "'legendArgs' must be NULL or a list whose names match the arguments of 'legend()'",

                      # lineArgs
                      if (!is.null(lineArgs)) {
                        is.list(lineArgs)
                      } else TRUE,
                      "'lineArgs' must be NULL or a list of named values that can be passed to 'abline()'",
                      
                      # textArgs
                      if (!is.null(textArgs)) {
                        is.list(textArgs)
                      } else TRUE,
                      "'textArgs' must be NULL or a list of named values that can be passed to 'text()'")
                      
    
  # Get the truth column
  truthCol <- attributes(x)$truthCol

  # Get the number of obs
  n <- nrow(x)

  # Make the call to par()
  if (is.null(parArgs)) {
    parArgs <- list()
  }
  defaultParArgs <- list(mar = c(5.5, 2, 2, 0.5), mgp = c(4, 1, 0))
  op <- do.call(par, blendArgs(defaultParArgs, parArgs))
                      
  # Get the jittered y-value
  set.seed(seed)
  y <- Smisc::linearMap(rnorm(n), R = c(-1, 1))

  # Get the value of the second level of the classes
  secondLevel <- attributes(x)$classNames[2]
  
  # Set pch and col values
  if (is.null(truthCol)) {
      
    pchVal <- pch[1]
    colVal <- col[1]
    
  }
  else {

    # Ensure we have two values for pch and col
    if (length(pch) == 1) {
      pch <- rep(pch, 2)
    }
    if (length(col) == 1) {
      col <- rep(col, 2)
    }
                        
    # Indicator of second truth class
    classInd <- x[,truthCol] == secondLevel

    # Set the pch
    pchVal <- rep(pch[1], n)
    pchVal[classInd] <- pch[2]

    # Set the col
    colVal <- rep(col[1], n)
    colVal[classInd] <- col[2]

  }

  # Transformed probabilities for plotting
  tr <- function(x) pbeta(x, scale, scale)
  trx <- tr(x$Prob)

  # Set up the default plotting parameters
  defaultPlotArgs <- list(x = trx,
                          y = y,
                          pch = pchVal,
                          col = colVal,
                          ylim = c(-2, 2),
                          xlim = range(trx),
                          frame.plot = TRUE,
                          axes = FALSE,
                          ylab = "",
                          xlab = paste("Pr(observation belongs to class '", secondLevel, "')", sep = ""))
  
  # Blend default with supplied arguments, giving preference to suppliedPlotArgs
  plotArgs <- list(...)
  finalPlotArgs <- blendArgs(defaultPlotArgs, plotArgs)

  # Determine the ylab that will be plotted
  if ("ylab" %in% names(plotArgs)) {
    ylab <- plotArgs$ylab
  }
  else {
    ylab <- "Random jitter"
q  }
  
  # Set ylab to "" so we don't write anything there
  finalPlotArgs$ylab <- ""
    
  # Make the plot
  do.call(plot, finalPlotArgs)

  # Choose the cex.lab.  Give local parms priority
  cex.lab <- 1
  if (!is.null(parArgs$cex.lab)) {
    cex.lab <- parArgs$cex.lab * 1.1
  }
  if (!is.null(finalPlotArgs$cex.lab)) {
    cex.lab <- finalPlotArgs$cex.lab * 1.1
  }
  
  # Add the y-axis label manually
  mtext(ylab, side = 2, line = 1, cex = cex.lab)
    
  # Add the axes if 'axes' wasn't set to TRUE
  if (!finalPlotArgs$axes) {

    xaxisPts <- seq(finalPlotArgs$xlim[1], finalPlotArgs$xlim[2], length = 9)
    axis(1, at = xaxisPts, labels = Smisc::padZero(round(qbeta(xaxisPts, scale, scale), 4), side = "right"), las = 2)
    
  }

  # Add the vertical bar for tau
  tau <- attributes(x)$optimalParms[["tau"]]
  defaultLineArgs <- list(v = tr(tau), col = "Black", lwd = 2, lty = 2)
  do.call(abline, blendArgs(defaultLineArgs, if (is.null(lineArgs)) list() else lineArgs))

  # Add in the text for tau
  if (is.null(textArgs)) {
    textArgs <- list()
  }

  # If x and y are in text args and are numeric, transform the x values
  if (all(c("x", "y") %in% names(textArgs))) {
    if (is.numeric(textArgs$x) & is.numeric(textArgs$y)) {
      textArgs$x <- tr(textArgs$x)
    }
  }
  
  defaultTextArgs <- list(x = tr(tau), y = max(finalPlotArgs$ylim), pos = 4,
                          labels = quote(substitute(paste(tau, " = ", k), list(k = tau))))
  do.call(text, blendArgs(defaultTextArgs, textArgs))

  # Add the legend if truthCol was provided
  if (!is.null(truthCol)) {

    if (is.null(legendArgs)) {
      legendArgs <- list()
    }

    # If x and y are in legend args and are numeric, transform the x values
    if (all(c("x", "y") %in% names(legendArgs))) {
      if (is.numeric(legendArgs$x) & is.numeric(legendArgs$y)) {
        legendArgs$x <- tr(legendArgs$x)
      }
    }

    # Specify the default legend arguments
    defaultLegendArgs <- list(x = "topleft",
                              legend = paste("True class is '", attributes(x)$classNames, "'", sep = ""),
                              pch = pch,
                              col = col)

    # Make the legend
    do.call(legend, blendArgs(defaultLegendArgs, legendArgs))

  }

  # restore original par settings
  par(op)
  
} # plot.LRCpred

# These are tested in "tests/testthat/test-listParNames.R"
listParNames <- function() {
    
  c("xlog", "ylog", "adj", "ann", "ask", "bg", "bty", "cex", "cex.axis", "cex.lab",
    "cex.main", "cex.sub", "cin", "col", "col.axis", "col.lab", "col.main", "col.sub",
    "cra", "crt", "csi", "cxy", "din", "err", "family", "fg", "fig", "fin", "font",
    "font.axis", "font.lab", "font.main", "font.sub", "lab", "las", "lend", "lheight",
    "ljoin", "lmitre", "lty", "lwd", "mai", "mar", "mex", "mfcol", "mfg", "mfrow", "mgp",
    "mkh", "new", "oma", "omd", "omi", "page", "pch", "pin", "plt", "ps", "pty", "smo", "srt",
    "tck", "tcl", "usr", "xaxp", "xaxs", "xaxt", "xpd", "yaxp", "yaxs", "yaxt", "ylbias")
  
}


# Create the union of the defaultArgs and the supplied args, ..., but supplied args get preference
# if there are two of the same name
blendArgs <- function(defaultArgs, suppliedArgs) {

  # Add in defaultArgs not already in suppliedArgs
  c(suppliedArgs, defaultArgs[setdiff(names(defaultArgs), names(suppliedArgs))])
  
}
