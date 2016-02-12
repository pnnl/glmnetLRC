##' Generic for \code{extract} method.
##'
##' @param object The object on which the generic operates
##' @param \dots Arguments passed to specific methods
##' 
##' @export
extract <- function (object, ...) {
  UseMethod("extract", object)
}

