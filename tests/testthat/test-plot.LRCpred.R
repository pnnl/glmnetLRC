context("Tests for plot.LRCpred()")

# Load the fitted model object
data(glmnetLRC_fit, package = "glmnetLRC")

# Get test data
data(testdata, package = "glmnetLRC")

# Make some predictions
new <- predict(glmnetLRC_fit, testdata, truthCol = "Curated_Quality")
new1 <- predict(glmnetLRC_fit, testdata, keepCols = 1:2)


pdf("test_plot.LRCpred.pdf")

test_that("plot.LRCpred() makes plots as expected", {

  plot(new)
  plot(new1)
    
  plot(new, pch = c(4, 2), col = c(3, 5), scale = 0.5, seed = 7,
       parArgs = list(las = 1), legendArgs = list(x = "topright"), textArgs = list(cex = 1.7),
       lineArgs = list(lty = 1), main = "test1", ylim = c(-5, 5))

  plot(new1, pch = 4, col = c(3, 5), scale = 0.5, seed = 8,
       parArgs = list(las = 1), legendArgs = list(x = "topright"), textArgs = list(cex = 1.7),
       lineArgs = list(lty = 1), main = "test2", frame.plot = FALSE)

  plot(new, col = 3, ylab = "try new")

  plot(new1, ylab = "try new1")

  plot(new, ylab = "")
  plot(new1, ylab = "")

})

test_that("plot.LRCpred() returns errors as expected", {

  expect_error(plot(new, pch = c(4, 2, 5)), "'pch' must be")

  expect_error(plot(new, col = numeric(0)), "'col' must be")

  expect_error(plot(new, scale = "yellow"), "'scale' must be")
               
  expect_error(plot(new, scale = 8), "'scale' must be")

  expect_error(plot(new, seed = TRUE), "'seed' must be")

  expect_warning(plot(new, parArgs = list(nonsense = "this")))

  expect_error(plot(new, parArgs = "nogo"), "'parArgs' must be")

  expect_error(plot(new, legendArgs = list(nonsense = "that")), "'legendArgs' must be")

  expect_error(plot(new, legendArgs = FALSE), "'legendArgs' must be")

  expect_error(plot(new, lineArgs = 7), "'lineArgs' must be")

  expect_warning(plot(new, lineArgs = list(nonsense = 7)))

  expect_error(plot(new, textArgs = TRUE), "'textArgs' must be")

  expect_warning(plot(new, textArgs = list(nonsense = 7)))
               
})

dev.off()
               
unlink("test_plot.LRCpred.pdf")
