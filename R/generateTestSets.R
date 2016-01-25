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


################################################################################
# Helper function to generate the testing sets for cross validation, LOO,
# etc.
################################################################################

generateTestSets <- function(truthLabels, cvFolds, cvReps, masterSeed, stratify) {

   # The number of observations
   n <- length(truthLabels)
  
   # If cvFolds is equal to N, we have L.O.O. cross validation, and it doesn't make
   # sense to replicate.  Set cvReps to 1 if cvFolds == n.  No stratification if
   # we're performing L.O.O.
   if (n == cvFolds) {
     cvReps <- 1
     stratify <- FALSE
   }
    
   ################################################################################ 
   # cvReps of cvFold cross validation, this also handles L.O.O.
   ################################################################################ 

   if (!stratify) {
       
     # A vector of seeds for cvReps
     seedVec <- createSeeds(masterSeed, cvReps)
     names(seedVec) <- paste("cvRep =", 1:cvReps)
  
     # Generate replicated partitions
     out <- lapply(seedVec, function(x) Smisc::parseJob(n, cvFolds, random.seed = x))
     
   }
     
   ################################################################################  
   # Stratified sampling to ensure proporational sampling of both levels of the
   # response
   ################################################################################
   else {

     # Create a mapping for each level of the truthLabels
     level1indexes <- which(truthLabels == levels(truthLabels)[1])
     level2indexes <- which(truthLabels == levels(truthLabels)[2])
     nlevel1 <- length(level1indexes)
     nlevel2 <- length(level2indexes)

     # Base requirement:  each training set must have at least one ob from each level.
     # The requirement implemented below is more stringent:  each training set will have at least
     # cvFolds - 1 obs of each level (i.e., each testing set has at least one ob from each level)
      
     # Now ensure that at least 1 ob from each level will be present in each test set
     if ((nlevel1 < cvFolds) | (nlevel2 < cvFolds)) {

       if (nlevel1 < cvFolds) {
         value <- levels(truthLabels)[1]
         nlevelval <- nlevel1
       }
       else {
         value <- levels(truthLabels)[2]
         nlevelval <- nlevel2
       }
        
       stop("There are ", nlevelval, " observations where truthLabels = '", value,
            "', which is smaller than the number of cvFolds = ", cvFolds, ".\n",
            "This makes it difficult to create stratified cross-validation partitions.\n",
            "You should decrease the size of 'cvFolds'.")
     }

     # A function for creating a proportional allocation of each level to each fold
     allocate <- function(twoSeeds) {

       # twoSeeds is a 2-vector, with a seed for level1 and level2

       # Get the allocation for level 1 & 2
       l1 <- Smisc::parseJob(nlevel1, cvFolds, random.seed = twoSeeds[1])
       l2 <- Smisc::parseJob(nlevel2, cvFolds, random.seed = twoSeeds[2])

       # Map the output to the true indexes in truthLabels
       l1mapped <- lapply(l1, function(x) level1indexes[x])
       l2mapped <- lapply(l2, function(x) level2indexes[x])

       # combine the two levels together. Initialize an output vector
       res <- vector(mode = "list", length = cvFolds)

       for (i in 1:cvFolds) {
         res[[i]] <- c(l1mapped[[i]], l2mapped[[i]])
       }

       # Check the results
       check <- sort(unlist(res))

       if (length(check) != n) {
         stop("Algorithm for generating stratified cross validation folds failed, incorrect length")
       }

       if (!all(check == 1:n)) {
         stop("Algorithm for generating stratified cross validation folds failed, incorrect indexes")            
       }

       # Return results
       return(res)
        
     } # allocate

     # Create two sets of seeds, twice the needed length
     seedVec <- createSeeds(masterSeed, 2 * cvReps)

     # Convert it to a list of 2-vectors
     seedList <- Smisc::df2list(as.data.frame(matrix(seedVec, ncol = 2)), out.type = "vector")
     
     # Now create multiple stratified partitions
     out <- lapply(seedList, allocate)
      
   } # else stratify
    
   return(out)
    
} # generateTestSets()

