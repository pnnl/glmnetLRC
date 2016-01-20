## Prediction function for the elastic net logistic regression classifier
##
## Prediction function for the elastic net logistic regression classifier
## for a variety of thresholds and calculate
## the associated loss
##
## Let \code{z} represent class denoted by the last level of \code{truthLabels}.
## Then the probability returned by \code{predict(glmnetFit, newData, type = 'response')}
## is the probability that the observations belong to class z.  If this probabilty
## exceeds \code{tau}, we will classify the observation as belonging to \code{z}.
##
## @author Landon Sego
##
## @param LRfit The glm fitted object (returned from \code{glm})
##
## @param XY_new A matrix or dataframe of new instances that match the
## predictors and corresponding response of binary truth labels
##
## @param lossMat A loss matrix of class \code{lossMat}, returned by
## \code{lossMatrix}
##
## @param tauVec A numeric sequence of threshold values for the binary
## classification.
##
## @param weight A numeric vector indicating the relative weight to ascribe
## to each row of \code{newData}
##
## @return A data frame containing \code{weightedSumLoss} (the sum of the
## product of the weights and the loss) and \code{sumWeights}
## (the sum of the weights) for each value of the threshold \code{tau}.  The
## loss (for \code{newData}) is given by \code{weightedSumLoss} divided by
## \code{sumWeights}.

################################################################################
# TESTS TO DO
#
# On prediction, what happens if extra columns are presented in newdata?
# 1) Usually prediction on fits only makes sense if the subspace where newdata
# lies is in the same subspace as the training data
# What happens if not all the predictors are in newdata?
# 2) Same
# What happens with missing data in newdata?
# What happens if column names are not provided in newdata?
################################################################################

predLoss_LRCbestsubsets <- function(LRfit, Xy_new, lossMat,
                                    tauVec = seq(0.1, 0.9, by = 0.1),
                                    weight = rep(1, nrow(Xy_new))) {

  # Extract truth labels and predictors from Xy_new
  truthLabels <- Xy_new[,ncol(Xy_new)]
  newPreds <- Xy_new[,-ncol(Xy_new)]

  # Check inputs
  stopifnot(inherits(LRfit, "glm"),

            is.factor(truthLabels),

            # Ensure we have a binary response
            length(levels(truthLabels)) == 2,

            is.numeric(tauVec),

            length(weight) == NROW(Xy_new))

  # Force the weight to resolve to avoid any issues with lazy evaluation
  force(weight)

  # For each lambda, make probabality predictions that the instance is an
  # element of the class with the largest factor level
  preds <- predict(LRfit, newPreds, type = "response")

  # Let Z be the last level of the response (Z = levels(truthLabels)[2])
  # The preds matrix returned by predict(LRfit, ...) is  P(x elem Z)
  # The rule will be:  If P(x elem Z) > tau --> x is assigned to Z
  # Create a logical matrix of whether x is assigned to Z for a given tau

  # Calculate the loss over the tau
  calcLossOverTau <- function(tau) {

    # x is the probabilty returned by predict(LRfit, ...), i.e.,
    # it is P(x elem Z)

    # Dichotomize the prediction for tau
    predLabels <- factor(preds > tau, levels = c(FALSE, TRUE),
                         labels = levels(truthLabels))

    # Calculate the loss
    cl <- calcLoss(truthLabels, predLabels, lossMat, weight = weight)

    return(c(list(tau = tau), cl))

  }

  # Calculate the weighted sum of the loss and the sum of the weights
  # (aggregated over the observations provided in 'newData') for
  # each combination of tau and lambda
  return(list2df(lapply(tauVec, calcLossOverTau)))

} # predLoss_LRCbestsubsets
