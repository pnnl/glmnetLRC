context("Verify glmnetLRC() performs correctly")

test_that("Final glmnet model matches manual fitting", {

  # Load the glmmnet library
  
    
  # Set the seed
  set.seed(20)
    
  # A response variable
  response <- factor(rep(c("a", "b"), each = 20))

  # Convert response to 0,1:  a = 0, b = 1
  y <- as.numeric(response) - 1

  # Create 2 predictors that depend somewhat on the response
  preds <- matrix(c(rnorm(40, 0, 0.5) + y, rnorm(40, 2, 0.5) + y),
                  ncol = 2, dimnames = list(NULL, c("x1", "x2")))

  # Set alpha
  alpha <- 0.7
  
  # Get a single fit of the data
  gfit <- glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3, cvReps = 2, standardize = FALSE)

  # Get optimal parms
  lambda <- print(gfit)[,"lambda"]

  # Get lambdaVec
  lambdaVec <- gfit$lambda

  # Fit the glmnet
  gfit1 <- glmnet(preds, response, family = "binomial", alpha = alpha, lambda = lambdaVec, standardize = FALSE)

  # Get the coefs with the optimal lambda
  c1 <- coef(gfit)
  names(c1) <- NULL
  c2 <- as.vector(coef(gfit1, s = lambda))

  # Create an objective function that should match results from glmnet
  f <- function(beta, n = 40, lambda = 1, alpha = 0.7) {
  
    # Name the regression parameters for easier readability
    beta0 <- beta[1]
    beta1 <- beta[2]
    beta2 <- beta[3]
  
    # The linear combination of regression parameters and predictors
    xb <- beta0 + beta1 * preds[,1] + beta2 * preds[,2]
  
    # The unpenalized binomial log-likelihood
    ll <- sum(y * xb - log(1 + exp(xb))) / n
  
    # The elastic-net penalty
    penalty <- lambda * (0.5 * (1 - alpha) * (beta1^2 + beta2^2) + alpha * (abs(beta1) + abs(beta2)))
  
    # The objective to minimize
    return(penalty - ll)
      
  } # f()

  # Fit the parameters manually
  c3 <- optim(c(-3, 1, 1), f, alpha = alpha, n = nrow(preds), lambda = lambda)$par

  # Tests
  expect_equal(c1, c2)
  expect_true(max(abs(c3 - c1)) < 0.001)

})


  # Create a vector of labels, simulating instances
  tClass <- factor(rep(letters[1:3], each = 5))
  pClass <- sample(tClass)
  weights <- rpois(15, 3)

  # Calculate the loss 
  noAggLoss <- calcLoss(tClass, pClass, lMat, aggregate = FALSE)
  AggLoss <- calcLoss(tClass, pClass, lMat, lossWeight = weights)

  # Manually calculate the loss
  manLoss <- double(15)
  
  for (i in 1:15) {

    if (((tClass[i] == "a") & (pClass[i] == "b")) |
        ((tClass[i] == "b") & (pClass[i] == "a")) |
        ((tClass[i] == "b") & (pClass[i] == "c")) |
        ((tClass[i] == "c") & (pClass[i] == "b"))) {
      manLoss[i] <- 1
    }
    else if (((tClass[i] == "a") & (pClass[i] == "c")) |
             ((tClass[i] == "c") & (pClass[i] == "a"))) {
      manLoss[i] <- 2
    }
  } # for

  # Checks
  expect_equal(noAggLoss$loss, manLoss)
  expect_equal(AggLoss$weightedSumLoss, as.vector(t(weights) %*% manLoss))
  expect_equal(AggLoss$sumWeights, sum(weights))
    
})


# Get some objects that both of the next tests will need

# Load the fitted model object
data(glmnetLRC_fit, package = "glmnetLRC")

# Get test data
data(testdata, package = "glmnetLRC")

# Make some predictions
new <- predict(glmnetLRC_fit, testdata, truthCol = "Curated_Quality")


test_that("calcLoss() remains unchanged from historical result", {

  # Set the loss matrix
  lM <- lossMatrix(c("good","good","poor","poor"),
                    c("good","poor","good","poor"),
                    c(     0,     1,     5,     0))

  # Calculate the loss
  loss <- with(new, calcLoss(Curated_Quality, PredictClass, lM))

  # Check agains the 'known' values of 21 and 99
  expect_equal(loss$weightedSumLoss, 21)
  expect_equal(loss$sumWeights, 99)

})

test_that("Confusion matrix methods from summary() are correct", {

  # Check the summary as well
  stats <- summary(new)

  # Manually calculate CM metrics
  sensitivity <- with(new, sum((PredictClass == "poor") & (Curated_Quality == "poor")) / sum(Curated_Quality == "poor"))
  specificity <- with(new, sum((PredictClass == "good") & (Curated_Quality == "good")) / sum(Curated_Quality == "good"))
  accuracy <- with(new, sum(PredictClass == Curated_Quality) / nrow(new))

  # Tests
  expect_equal(stats["sensitivity",], sensitivity)
  expect_equal(stats["specificity",], specificity)
  expect_equal(stats["accuracy",], accuracy)
  
})


test_that("Counting from summary() remains unchanged from historical result", {

  # Make predictions without the truth column
  new1 <- predict(glmnetLRC_fit, testdata)

  # Remove blank spaces in the summary
  out <- gsub(" ", "", summary(new1))

  # Tests
  expect_equal(out[1,], "good:35")
  expect_equal(out[2,], "poor:64")
    
})
