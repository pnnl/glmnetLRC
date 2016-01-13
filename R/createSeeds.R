################################################################################
# Helper function to ensure repeatability of seeds across parallel threads
################################################################################

createSeeds <- function(masterSeed, cvReps) {

  # Create the vector of seeds that will be used in the parallel call
  set.seed(masterSeed)
  seedVec <- unique(as.integer(runif(cvReps * 2, min = 1, max = cvReps * 10)))

  # Make sure the length of seedVec is >= cvReps.  If not, add more seeds
  i <- 0

  while ((length(seedVec) < cvReps) & (i < 20)) {

    seedVec <- unique(c(seedVec,
                        as.integer(runif((cvReps - length(seedVec)) * 10,
                                         min = 1, max = cvReps * 10))))

    i <- i + 1

  }

  if (i >= 20) {
    stop("Unique seed vector was not created")
  }
    

  # Randomly select a vector of seeds from the unique set
  return(sample(seedVec, cvReps))

} # createSeeds

