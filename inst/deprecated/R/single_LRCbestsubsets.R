## Train best subsets logistic regression classifier for a single cross validation
## run.  A helper function for LRCbestsubsets()

single_LRCbestsubsets <- function(Xy,
                                  lossMat,
                                  weight,
                                  tauVec,
                                  cvFolds,
                                  seed,
                                  n,
                                  verbose,
                                  ...){

  # ... are arguments passed to best glm and glm (other than 'family') should check this

  # Function to train and test over a single CV fold
  trainTest <- function(testSet) {

    # testset:  indexes in 1:n of the data set that will be used to test

    # Find the complement of the testSet
    trainSet <- sort(setdiff(1:n, testSet))

    # Determine whether there are a sufficient number of binary (0 & 1) cases
    # in the training set
    tab <- table(Xy[trainSet,NCOL(Xy)])
    
    if (length(tab) < 2) {

      # Determine which type was missing
      missingType <- setdiff(levels(Xy[,NCOL(Xy)]), names(tab))
      
      cat("For seed '", seed, "', there were no observations in the training data set with\n",
              "response '", missingType, "'. No model was fit for this cross-validation replicate.", sep = "")

      return(NULL)
      
    }

    # Train the best subsets logistic regression
    bestSub <- bestglm::bestglm(Xy[trainSet,], weights = weight[trainSet],
                                family = binomial, ...)$BestModel

##     bestPred <- rownames(print(bestSub))[-1]

##     # Select the terms of the best subsets
##     bestDataMatrix <- dataMatrixTrain[, c(bestPred, truthLabelName)]

##     # Now fit the best logistic regression model
##     LRfit <- glm(as.formula(paste(truthLabelName, "~ .")),
##                  data = bestDataMatrix, weight = weight[trainSet],
##                  family = "binomial", ...)

    # Now test it
    out <- predLoss_LRCbestsubsets(bestSub,
                                   Xy[testSet,],
                                   lossMat,
                                   tauVec = tauVec,
                                   weight = weight[testSet])

    return(out)

  } # trainTest


  # Generate the test folds

  # If we have LOO:
  if (n == NROW(Xy)) {

    # The randomization doesn't matter in this case--but we'll throw it in
    # for completeness since it gets included in the list this function returns
    testFolds <- Smisc::parseJob(n, n, random.seed = seed)
  }
  # For non-LOO cross validation
  else {
    testFolds <- Smisc::parseJob(n, cvFolds, random.seed = seed)
  }

  # Test/train over over the folds
  completeTest <- Smisc::list2df(lapply(testFolds, trainTest))

  # Now summarize the loss over the cv folds, with a loss value for each
  # tau combination for a given seed
  dfData <- list2df(plyr::dlply(completeTest,

                                .variables = c('tau'),

                                .fun = function(x){

                                  # x = K x K data.frame of values for the K folds with
                                  # same (alpha, lambda, tau, seed) parameter values.
                                  Eloss <- sum(x$weightedSumLoss) / sum(x$sumWeights)
 
                                  return(list('ExpectedLoss' = Eloss,
                                              'tau' = unique(x$tau)))
                                }),

                     row.names = NULL)


  if (any(is.na(dfData))) {
    warning("Unexpected NA values are present in the cross\n",
             "validation results for replicate seed = ", seed, "\n")
  }


  # Searching for the minimum by sorting. Smaller expected loss is preferred
  # In the event of a tie, smaller sqErrorTau is prferred (tau closer to 0.5)
  dfData$sqErrorTau <- (dfData$tau - 0.5)^2
  gridMinimum <- Smisc::sortDF(dfData, ~ExpectedLoss + sqErrorTau)[1,]

  # Add in the seed
  gridMinimum$cvRepSeed <- seed

  # Return the optimal tau for this particular seed
  return(gridMinimum[,c("cvRepSeed", "tau", "ExpectedLoss")])

} # single_LRCbestsubsets

