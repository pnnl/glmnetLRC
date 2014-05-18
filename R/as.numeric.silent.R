# Function to convert a vector to numeric (without producing errors or warnings)
as.numeric.silent <- function(x) {

  # Set warnings to errors
  op <- options(warn = 2)

  # If an error results, the data aren't numeric
  if (class(x.num <- try(as.numeric(x), silent = TRUE)) == "try-error")
    x.num <- x

  # Restore the previous setting
  options(op)

  return(x.num)
    
} # as.numeric.silent()
