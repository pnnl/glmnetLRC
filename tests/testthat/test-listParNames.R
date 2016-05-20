context("Test the par names returned by 'listParNames()'")

test_that("'listParNames()' returns current list of parnames", {

  # Get the true par names
  dev.new()
  trueParNames <- names(par())
  dev.off()

  # Get the ones we have listed
  lParNames <- listParNames()
  
  expect_equal(length(lParNames), length(trueParNames))
  expect_true(setequal(lParNames, trueParNames))
  
})
