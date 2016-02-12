# Builds the vignette in the glmnetLRC package
# Usage:  ./build.sh
R CMD Sweave glmnetLRC.Rnw
pdflatex glmnetLRC
bibtex glmnetLRC
pdflatex glmnetLRC
pdflatex glmnetLRC
cp glmnetLRC.pdf ../inst/doc/.
rm glmnetLRC.aux
rm glmnetLRC.bbl
rm glmnetLRC.blg
rm glmnetLRC.log

