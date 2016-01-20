
# For building during development only
## require(pnlStat)
## require(roxygen2)
## rma()
## unloadNamespace("lrc")
## roxygenize("~/rp/lrc", clean = TRUE)
## system("R CMD INSTALL  ~/rp/lrc")


require(lrc)

################################################################################
# Test example (upon which to base the vignette) begins here
################################################################################

# Load the Mojave data
data(Mojave)

# Here we select the predictor variables
predictors <- Mojave[,-c(1,2,11)]

# And the response (presence/absence of cheat grass)
cheat <- Mojave$cheatGrass

# Specify the loss matrix.
# The "1" class is the target of interest (indicating the presence of cheatgrass).
# The penalty for missing cheat grass is 2, while the penalty for predicting it
# falsely is 1.
lM <- lossMatrix(c("0","0","1","1"),
                 c("0","1","0","1"), 
                 c(0,   1,  2,  0))
print(lM)

# Train the elastic net classifier
LRCbestsubsets_fit <- LRCbestsubsets(cheat, predictors, lM, cvReps = 100,
                                     cvFolds = 5, cores = 7)


#save(LRCbestsubsets_fit, file = "~/rp/lrc/data/LRCbestsubsets_fit.RData")

#LRCbestsubsets_fit <- loadObject("~/rp/lrc/data/LRCbestsubsets_fit.RData")

# Demonstrate the various methods (print, summary, plot, coef)
print(LRCbestsubsets_fit)

o <- print(LRCbestsubsets_fit)
o

summary(LRCbestsubsets_fit)

openDevice("~/tmp/testPackage.pdf")
plot(LRCbestsubsets_fit)
dev.off()

coef(LRCbestsubsets_fit)

# Calculate performance of the final model on all the training data
out <- predict(LRCbestsubsets_fit, cbind(predictors, cheat),
               truthCol = "cheat", keepCols = 1:3)

head(out)
summary(out)

