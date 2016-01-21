# A test to ensure glmnet is functioning per the documentation at
# https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet_beta.html
library(glmnet)

set.seed(20)

# A response variable
response <- factor(rep(c("a", "b"), each = 20))

# Convert response to 0,1:  a = 0, b = 1
y <- as.numeric(response) - 1

# Create 2 predictors that depend somewhat on the response
preds <- matrix(c(rnorm(40, 0, 0.5) + y, rnorm(40, 2, 0.5) + y),
                ncol = 2, dimnames = list(NULL, c("x1", "x2")))

# Fit the glmnet with alpha = 0.7
gfit <- glmnet(preds, response, family = "binomial", alpha = 0.7)

# Create an objective function that should match results from glmnet
f <- function(beta, lambda = 1, alpha = 0.7) {

  # Name the regression parameters for easier readability
  beta0 <- beta[1]
  beta1 <- beta[2]
  beta2 <- beta[3]

  # The linear combination of regression parameters and predictors
  xb <- beta0 + beta1 * preds[,1] + beta2 * preds[,2]

  # The unpenalized binomial log-likelihood
  ll <- sum(y * xb - log(1 + exp(xb)))

  # The elastic-net penalty
  penalty <- lambda * (0.5 * (1 - alpha) * (beta1^2 + beta2^2) + alpha * (abs(beta1) + abs(beta2)))

  # The objective to minimize
  return(penalty - ll)
    
} # f()

# Compare the coeffficients from glmnet and our manual fitting, using a lambda from
# roughly in the middle of the lambda sequence.  Why are they so different?
coef(gfit, s = 0.072, exact = TRUE)
optim(c(-5.2, 1.3, 1.8), f, lambda = 0.072)$par

# Just for kicks, if I increase lambda by a factor of 20, they begin to resemble
# what glmnet returns
optim(c(-5.2, 1.3, 1.8), f, lambda = 20 * 0.072)$par

# Check the objective function against glm
d <- data.frame(r = response, x1 = preds[,1], x2 = preds[,2])
glmfit <- glm(r ~ x1 + x2, data = d, family = "binomial")

# Compare glm coefs to our manual fitting with lambda = 0. Very comparable:
coef(glmfit)
optim(c(-5.2, 1.3, 1.8), f, lambda = 0)$par

