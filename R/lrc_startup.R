.onAttach <- function(libname, pkgname) {

  banner.text <- paste("\n\nThe package vignette can be found at ",
                       path.package(package="lrc"), "/doc/glmnetLRC.pdf\n",
                       sep="")
  
  citation.text <- paste("\nPlease cite the following reference:",
                         "\nAmidan BG, Orton DJ, LaMarche BL, et al. 2014.",
                         "\nSignatures for Mass Spectrometry Data Quality.",
                         "\nJournal of Proteome Research. 13(4), 2215-2222.\n", sep = "")

  packageStartupMessage(banner.text)
  packageStartupMessage(citation.text)
  
}
