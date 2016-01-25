## Prepare data for fitting glmnetLRC model
##
## Removing missing observations and columns with too much missing data
## A helper function for \code{\link{glmnetLRC}}

dataPrep <- function(truthLabels,
                     predictors,
                     weight,
                     naFilter,
                     verbose) {

  
  # If truthLabels is not a factor, make it one
  if (!is.factor(truthLabels))
    truthLabels <- as.factor(truthLabels)

  # Remove predictors that have fewer than naFilter% observations
  # TRUE means that predictor has enough data
  selPreds <- apply(predictors, 2,
                    function(x) {
                     (sum(!is.na(x)) / NROW(predictors)) > naFilter
                    })

  # If all predictors are missing too many obs
  if (!any(selPreds))
    stop("All predictors had more than ", naFilter * 100,
         "% of their observations being missing")

  # If some of the predictors are missing too many obs
  if (!all(selPreds)) {

    if (verbose)
      cat("The following predictors were removed because more than\n",
          naFilter * 100, "% of their observations were missing:\n",
          "'", paste(colnames(predictors)[!selPreds], collapse = "', '"), "'\n",
          sep = "")
    
    predictors <- predictors[,selPreds]

  }

  # Identify the obs that don't have any missing predictors or responses
  cc <- complete.cases(cbind(truthLabels, predictors))

  # If there are no complete cases
  if (!any(cc)) {
    stop("There are no complete rows of data remaining for analysis")
  }

  # Remove obs that don't have complete cases
  predictors <- predictors[cc,]
  truthLabels <- truthLabels[cc]
  weight <- weight[cc]

  # Message about complete cases
  if (verbose & (!all(cc)))
    cat(sum(!cc), "observations (rows) were removed because one or more of",
        "their values were missing\n")
    
  if (verbose)
    cat(sum(cc), "observations are available for fitting the glmnetLRC model\n")

  # Return the data
  return(list(predictors = predictors,
              truthLabels = truthLabels,
              weight = weight))

} # dataPrep
