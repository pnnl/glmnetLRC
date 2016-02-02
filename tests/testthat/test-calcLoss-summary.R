context("Verify calcLoss() and summary() perform correctly")

test_that("calcLoss() is equivalent to manual loss calculation", {

  # Create a loss matrix
  lMat <- lossMatrix(rep(letters[1:3], 3), rep(letters[1:3], each = 3),
                     c(0, 1, 2, 1, 0, 1, 2, 1, 0))

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
