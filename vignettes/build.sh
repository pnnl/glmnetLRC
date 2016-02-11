# Builds the vignette in the glmnetLRC package
# Usage:  ./build.sh
R CMD Sweave glmnetLRC.Rnw
pdflatex glmnetLRC
bibtex glmnetLRC
pdflatex glmnetLRC
pdflatex glmnetLRC
