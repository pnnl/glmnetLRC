context("Verify calcLoss() and summary() perform correctly")

# Create a vector of labels, simulating instances
tClass <- factor(rep(letters[1:3], each = 5))

# Get a seed
seed <- rpois(1, 200)

# Parse the training sets
out <- generateTestSets(tClass, 3, 4, seed, FALSE)


# Tests
test_that("There should be two elements in the list, one for each of the 4 cvReps", {

  expect_equal(length(out), 4)

})

test_that("Names of list elements are as expected", {

  expect_equal(names(out), paste("cvRep =", 1:4))
  
})

test_that("Each vector of each sublist should have 5 elements", {
    
  expect_true(all(unlist(lapply(unlist(out, recursive = FALSE), length)) == 5))

})


test_that("Each sublist has a set of indexes equal to 1:N", {

  expect_true(all(unlist(lapply(out, function(x) all(sort(unlist(x)) == 1:length(tClass))))))
    
})

test_that("Cannot have more than 2 levels for stratified", {

  expect_error(generateTestSets(tClass, 2, 1, seed, TRUE), "There must be 2 levels in 'truthLabels'")
                                
})


test_that("Check for each training set having at least one observation from each level is working," {

   ntClass <- factor(rep(letters[1:2], each = 3))    

   expect_error(generateTestSets(ntClass, 4, 4, seed, TRUE), "This makes it difficult to create stratified")
    
})

test_that("Stratified splitting remains unchanged from historical code", {

   vals <- factor(rep(c("a","b"), 3))

   out <- unlist(generateTestSets(vals, 2, 1, 1, TRUE))
   names(out) <- NULL

   expect_equal(out, c(3, 5, 2, 6, 1, 4))
   
})

test_that("Checking L.O.O. allocation", {

  out <- generateTestSets(tClass, length(tClass), 1, FALSE)
  out1 <- sort(unlist(out))
  names(out1) <- NULL

  expect_equal(out1, 1:15)
  expect_equal(length(out), 1)
  expect_equal(length(unlist(out, recursive = FALSE)), 15)
  expect_true(all(unlist(lapply(unlist(out, recursive = FALSE), length)) == 1))
    
}

