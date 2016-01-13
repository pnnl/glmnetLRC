##' Logistic regression classification (LRC) with an arbitrary loss function
##'
##' Logistic regression classification (LRC) with an arbitrary loss function
##' and variable selection via elastic net (glmnet)
## or best subsets.
##'
## \pkg{lrc} extends the \pkg{glmnet} and \pkg{bestglm}
##' \pkg{lrc} extends the \pkg{glmnet} 
## packages for training elastic net or best subset LRCs with a user-specified
##' package for training elastic net LRCs with a user-specified
##' discrete loss function.
##' 
##' There are a handful of functions you'll need (along with their associated methods):
##' \code{\link{lossMatrix}}, \code{\link{LRCglmnet}}, 
##' and \code{\link{predict.LRCglmnet}}
##' 
## There are a handful of functions you'll need (along with their associated methods):
## \code{\link{lossMatrix}}, \code{\link{LRCglmnet}} or \code{\link{LRCbestsubsets}},
## and \code{\link{predict.LRCglmnet}} or \code{\link{predict.LRCbestsubsets}}.
##'
##' @author Landon Sego, Alex Venzin, John Ramey
##' @docType package
##' @name lrc-package
##' @rdname lrc-package
NULL
