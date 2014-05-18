# This function writes the value of variables (with labels) to the window

pvar <- function(..., digits = NULL, abbrev = NULL, verbose = TRUE) {

  # Grab the objects into a list
  vars <- list(...)
  vnames <- as.character(substitute(list(...)))[-1]

  # If a single list was provided
  if (length(vars) == 1) {
    if (is.list(vars[[1]])) {

      vars <- vars[[1]]
      vnames <- names(vars)

    }
  }

  # If an element of the list is NULL, replace it with a text string
  vars <- lapply(vars,
                 function(x) {
                   if (is.null(x)) 
                     return("NULL")
                   else
                     return(x)})

  # Get length of the whole list
  len <- length(vars)

  # Make abbreviations
  if (!is.null(abbrev)) {
    for (i in 1:len) {
      if (is.character(vars[[i]]))
        vars[[i]] <- substr(vars[[i]], 1, abbrev)
    }
  }

  # Truncate to desired digits
  if (!is.null(digits)) {
    for (i in 1:len) {
      if (is.numeric(vars[[i]]))
        vars[[i]] <- round(vars[[i]], digits)
    }
  }

  # Collapse the text into a single string
  out <- paste(paste(vnames, lapply(vars, paste, collapse=", "),
                     sep = " = "), collapse = "; ")

  if (verbose)
    cat(out, "\n")

  invisible(out)
  
} # end pvar()
