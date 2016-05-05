##' Generics for \code{extract} and \code{missingpreds} methods
##'
##' @aliases missingpreds
##' 
##' @usage extract(object, ...)
##' missingpreds(object, ...)
##' 
##' @param object The object on which the generic operates
##' @param \dots Arguments passed to specific methods
##'
##' @rdname generics
##' 
##' @export
extract <- function (object, ...) {
  UseMethod("extract", object)
}

##' @export
missingpreds <- function (object, ...) {
  UseMethod("missingpreds", object)
}
