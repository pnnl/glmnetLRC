context("Verify glmnetLRC() performs correctly")

# Load the glmmnet library
if (require(glmnet)) {
  stop("The 'glmnet' package must be installed for this test")
}
    
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
gfit <- glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                  cvReps = 2, standardize = FALSE, estimateLoss = TRUE)

test_that("0-1 matrix is created as expected", {

   m <- lossMatrix(c("a", "a", "b", "b"), c("a", "b", "a", "b"), c(0, 1, 1, 0))

   expect_equal(gfit$lossMat, m)

})

test_that("Order of loss matrix specification doesn't make a difference", {

   gfitNewLoss <- glmnetLRC(response, preds,
                            lossMat = lossMatrix(c("a", "b", "a", "b"), c("b", "a", "a", "b"), c(1, 1, 0, 0)),
                            alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                            cvReps = 2, standardize = FALSE, estimateLoss = TRUE)
   
   expect_equal(coef(gfitNewLoss), coef(gfit))

   # Get the printed objects
   g1 <- print(gfit)
   g2 <- print(gfitNewLoss)
   
   expect_equal(g1, g2)
        
})


test_that("Final glmnet model matches manual fitting", {

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


test_that("Testing intercept only model", {

  # Get a single fit of the data
  gfit <- glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3, cvReps = 2, standardize = FALSE, intercept = FALSE)

  # Get optimal parms
  lambda <- print(gfit)[,"lambda"]

  # Get lambdaVec
  lambdaVec <- gfit$lambda

  # Fit the glmnet
  gfit1 <- glmnet(preds, response, family = "binomial", alpha = alpha, lambda = lambdaVec, standardize = FALSE, intercept = FALSE)

  # Get the coefs with the optimal lambda
  c1 <- coef(gfit)
  names(c1) <- NULL
  c2 <- as.vector(coef(gfit1, s = lambda))

  expect_equal(c2[1], 0)

  c2 <- c2[-1]

  # Create an objective function that should match results from glmnet
  f <- function(beta, n = 40, lambda = 1, alpha = 0.7) {
  
    # Name the regression parameters for easier readability
    beta1 <- beta[1]
    beta2 <- beta[2]
  
    # The linear combination of regression parameters and predictors
    xb <- beta1 * preds[,1] + beta2 * preds[,2]
  
    # The unpenalized binomial log-likelihood
    ll <- sum(y * xb - log(1 + exp(xb))) / n
  
    # The elastic-net penalty
    penalty <- lambda * (0.5 * (1 - alpha) * (beta1^2 + beta2^2) + alpha * (abs(beta1) + abs(beta2)))
  
    # The objective to minimize
    return(penalty - ll)
      
  } # f()

  # Fit the parameters manually
  c3 <- optim(c(0.5, 0), f, alpha = alpha, n = nrow(preds), lambda = lambda)$par

  # Tests
  expect_equal(c1, c2)
  expect_true(max(abs(c3 - c1)) < 0.001)

})

test_that("Errors for argument checking are working as expected", {

  expect_error(glmnetLRC(rep(1,2,each=20), preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                         cvReps = 1.5, standardize = FALSE, estimateLoss = TRUE),
               "'truthLabels' must be a factor")
    
  expect_error(glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                         cvReps = 1.5, standardize = FALSE, estimateLoss = TRUE),
               "'cvReps' must be an integer")

  expect_error(glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                         cvReps = 2, stadardize = FALSE, estimateLoss = TRUE),
               "following do not match the arguments in glmnet")
  
  expect_error(glmnetLRC(response, preds, alphaVec = alpha, tauVec = 0.5, cvFolds = 3,
                         cvReps = 2, standardize = FALSE, family = "poisson", estimateLoss = TRUE),
               "are controlled and should not be supplied")


})
