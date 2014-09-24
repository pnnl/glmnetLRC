
# detach and reinstall the package
require(roxygen2)
unloadNamespace("glmnetLRC")

roxygenize("~/Shares/SQM/RPackages/SQM/trunk/glmnetLRC")
system("R CMD INSTALL  ~/Shares/SQM/RPackages/SQM/trunk/glmnetLRC")
require(glmnetLRC)
help.start()

# Load the VOrbitrap Shewanella QC data
data(traindata)
# Here we select the predictor variables
predictors <- as.matrix(traindata[,9:96])

# The logistic regression model requires a binary response
# variable. We will create a factor variable from the 
# Curated Quality measurements.

response <- factor(traindata$Curated_Quality,
                   levels = c("good", "poor"),
                   labels = c("good", "poor"))

# Specify the loss matrix. The "poor" class is the target of interest.
# The penalty for misclassifying a "poor" item as "good" results in a
# loss of 5.
lM <- lossMatrix(c("good","good","poor","poor"),
                 c("good","poor","good","poor"), 
                 c(     0,     1,     5,     0))

# Train the elastic net classifier
lassoModel <- trainGLR(response, predictors, lM, alphaVec = 1, tauVec = c(0.4, 0.5),
                       cores = 7, cvReps = 14)

print(lassoModel)
plot(lassoModel)


# Load the new observations
data(testdata)

# Use the trained model to make predictions about
# new observations for the response variable.

out <- predict(lassoModel, testdata, truthCol = "Curated_Quality",
               keepCols = 1:3)

head(out)

summary(out)

