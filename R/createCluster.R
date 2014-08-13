# Helper function for LRCbestsubsets and LRCglmnet for creating the cluster
# that will be used to parallelize the cross validation replicates

createCluster <- function(cvReps, masterSeed, cluster, cores) {

  ################################################################################
  # Set up cluster
  ################################################################################

  if (!is.null(cluster)) {
    if (!inherits(cluster, "cluster"))
      stop("'cluster' must inhererit the 'cluster' class")
  }

  # If the cluster is to be local, simply give the number of cores. If the
  # cluster is distributed, then a vector of nodeNames and a corresponding
  # vector denoting the number of cpus to use on each node must be supplied
  if (is.null(cluster)) {

   nCores <- detectCores()

    if (nCores < cores) {

      warning("Number of requested cores exceeds the number available on the host (",
              nCores, ")\n")

    }

    if (cvReps < cores) {

      warning("Number of cross validation replicates is less than the number of\n",
              "requested cores.  Setting number of cores to ", cvReps)

      cores <- cvReps

    }

    cl <- makeCluster(cores)

  } else {

    # Cluster has been provided
    cl <- cluster

  }

  ################################################################################
  # Train via cross validation
  ################################################################################

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

  # Randomly select a vector of seeds from the unique set
  seedVec <- sample(seedVec, cvReps)

  # Return the cluster
  return(list(cluster = cl, seedVec = seedVec))

} # createCluster
