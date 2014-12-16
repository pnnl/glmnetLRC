# Helper function for LRCbestsubsets and LRCglmnet for creating the cluster
# that will be used to parallelize the cross validation replicates

createCluster <- function(cvReps, cluster, cores) {

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

   nCores <- parallel::detectCores()

    if (nCores < cores) {

      warning("Number of requested cores exceeds the number available on the host (",
              nCores, ")\n")

    }

    if (cvReps < cores) {

      warning("Number of cross validation replicates is less than the number of\n",
              "requested cores.  Setting number of cores to ", cvReps)

      cores <- cvReps

    }

    cl <- parallel::makeCluster(cores)

  } else {

    # Cluster has been provided
    cl <- cluster

  }

  # Return the cluster
  return(cl)

} # createCluster
