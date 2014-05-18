# Function for converting all factors to characters in a data frame

factor2character <- function(dframe) {

  if (!is.data.frame(dframe))
    stop("'", deparse(substitute(dframe)), "' must be a dataframe.\n")

  for (cname in colnames(dframe)) {

    if (is.factor(dframe[,cname]))
      dframe[,cname] <- as.character(dframe[,cname])
    
  }

  return(dframe)

} # factor2character
