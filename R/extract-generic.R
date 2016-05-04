## Generics for \code{extract} and \code{missingpreds} methods
##
## @param object The object on which the generic operates
## @param \dots Arguments passed to specific methods
##
##' @export
extract <- function (object, ...) {
  UseMethod("extract", object)
}

##' @export
missingpreds <- function (object, ...) {
  UseMethod("missingpreds", object)
}



