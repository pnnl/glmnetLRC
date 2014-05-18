##' Build a loss matrix
##'
##' Build an arbitrary loss matrix for discrete classification
##'
##' This function checks the inputs and binds the three
##' arguments columnwise into a dataframe.
##' 
##' @export
##'
##' @rdname lossMatrix
##'
##' @author Landon Sego
##'
##' @param truthLabels character vector of truth labels
##' 
##' @param predLabels character vector of corresponding predicted labels,
##' which must be the same length as \code{truthLabels}
##'
##' @param lossValues numeric vector of corresponding loss values, which must
##' be the same length as \code{truthLabels} and \code{predLabels}.
##' 
##' @return An object of class \code{lossMat}: a dataframe that contains
##' all the information of the loss matrix to be used by \code{calcLoss}
##'
##' @examples
##' 
##' # A 2x2 symmetric loss matrix
##' lossMatrix(c("a","a","b","b"), c("a","b","a","b"), c(0, 1, 5, 0))
##'
##' # An unbalanced loss matrix (with a missing element)
##' lossMatrix(c("a","a","b"), c("a","b","b"), c(0, 1, 0))
##'
##' # A 3x2 asymmetric loss matrix
##' lossMatrix(rep(letters[1:3], each = 2), rep(letters[4:5], 3),
##'            c(0, 3, 2, 0, 1, 0))

lossMatrix <- function(truthLabels, predLabels, lossValues) {

  # Check inputs
  stopifnot(is.character(truthLabels),
            is.character(predLabels),
            is.numeric(lossValues),
            length(truthLabels) == length(predLabels),
            length(truthLabels) == length(lossValues))

  # Assemble into a dataframe
  out <- data.frame(truthLabels = truthLabels, predLabels = predLabels,
                    loss = lossValues)

  # Verify we don't have any duplicate pairings of truth and predicted values
  if (any(duplicated(out[,c("truthLabels","predLabels")])))
    stop("At least one pair of truth and predicted labels is duplicated")

  # Assign the class
  class(out) <- c("lossMat", class(out))

  # Return the loss matrix
  return(out)
  
} # lossMatrix

##' @rdname lossMatrix
##' @method print lossMat
##' @param x An object of class \code{lossMat}
##' @param \dots Additional arguments to \code{print.default}
##' @export

print.lossMat <- function(x, ...) {

  # Print the loss matrix as a matrix...

  # Get the unique truth and predicted labels
  truthL <- levels(x$truthLabels)
  predL <- levels(x$predLabels)

  # A matrix from
  m <- matrix(rep(NA, length(truthL) * length(predL)),
              nrow = length(truthL), dimnames = list(truthL, predL))

  # Assign values to the matrix
  for (tL in truthL) {
    for (pL in predL) {
      if (length(y <- x[x$truthLabels == tL & x$predLabels == pL, "loss"]))
        m[tL, pL] <- y
    }
  }

  # Add in prefixes
  rownames(m) <- paste("Truth", rownames(m), sep = ".")
  colnames(m) <- paste("Predicted", colnames(m), sep = ".")
  
  # Print the matrix
  print(m, ...)
  
} # print.lossMat

