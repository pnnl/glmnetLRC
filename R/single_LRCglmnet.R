## Train an elastic net logistic regression classifier for a single cross validation
## run.  A helper function for LRCglmnet()

single_LRCglmnet <- function(truthLabels,
                             predictors,
                             lossMat,
                             weight,
                             alphaVec,
                             tauVec,
                             cvFolds,
                             seed,
                             n,
                             verbose){

  # Function to train and test over the CV folds
  trainTest <- function(testSet, a = 1, lambdaV = NULL) {

    # testset:  indexes in 1:n of the data set that will be used to test
    # a is alpha
    # lambdaV is a vector of lambda values

    # Find the complement of the testSet
    trainSet <- sort(setdiff(1:n, testSet))

    # Train the elastic net regression
    glmnetFit <- glmnet::glmnet(predictors[trainSet,],
                                truthLabels[trainSet],
                                weights = weight[trainSet],
                                family = "binomial",
                                lambda = lambdaV,
                                alpha = a)

    # Now test it
    out <- predLoss_LRCglmnet(glmnetFit, predictors[testSet,], truthLabels[testSet],
                              lossMat, tauVec = tauVec, weight = weight[testSet])

    return(out)

  } # trainTest

  # Run the cross validation for a particular alpha
  cvForAlpha <- function(alpha, tFold) {

    # alpha is a scalar
    # tFold is list of training indexes--the output of parseJob()

    if (verbose)
      Smisc::pvar(alpha)

    # Get the lambdaVec for this particular alpha using all the data
    lambdaVec <- glmnet::glmnet(predictors, truthLabels, weights = weight,
                                family = "binomial", alpha = alpha)$lambda

    # Now train/test over all the cv folds
    testAll <- Smisc::list2df(lapply(tFold, trainTest, a = alpha,
                                     lambdaV = lambdaVec))


    # Add in the alpha
    testAll$alpha <- alpha


    return(testAll)

  } # cvForAlpha

  # Generate the test folds
  testFolds <- Smisc::parseJob(n, cvFolds, random.seed = seed)

  # Test/train over the vector of alphas
  completeTest <- Smisc::list2df(lapply(alphaVec, function(x) cvForAlpha(x, testFolds)))

  # Now summarize the loss over the cv folds, with a loss value for each
  # alpha, lambda, and tau combination for a given seed
  dfData <- Smisc::list2df(plyr::dlply(completeTest,

                                       .variables = c('alpha', 'lambda','tau'),

                                       .fun = function(x){

                                          # x = K x K data.frame of values for the K folds with
                                          # same (alpha, lambda, tau, seed) parameter values.
                                          Eloss <- sum(x$weightedSumLoss) / sum(x$sumWeights)

                                          return(list('ExpectedLoss' = Eloss,
                                                      'alpha' = unique(x$alpha),
                                                      'tau' = unique(x$tau),
                                                      'lambda' = unique(x$lambda)))
                                        }),

                           row.names = NULL)


  if (any(is.na(dfData)))
    warning("Unexpected NA values are present in the cross\n",
             "validation results for replicate seed = ", seed, "\n")


  # Searching for the minimum by sorting. Smaller expected loss is preferred
  # In the event of a tie, smaller sqErrorTau is prferred (tau closer to 0.5)
  # If still tied, larger values of lambda are prefered because they reduce the
  # number of predictors to create a more parsimonous model with fewer predictors
  dfData$sqErrorTau <- (dfData$tau - 0.5)^2
  gridMinimum <- Smisc::sortDF(dfData, ~ ExpectedLoss + sqErrorTau - lambda)[1,]

  # Add in the seed
  gridMinimum$seed <- seed

  # Return the optimal lambda, tau, and alpha for this particular seed
  return(gridMinimum[,c("seed", "alpha", "lambda", "tau", "ExpectedLoss")])

} # single_LRCglmnet

