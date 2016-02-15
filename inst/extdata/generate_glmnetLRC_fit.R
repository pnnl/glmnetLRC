## This code generates the 'glmnetLRC_fit' object stored in
## 'PACKAGE_DIR/data/glmnetLRC_fit.RData' and used in the
## glmnetLRC() example.

library(glmnetLRC)

data(traindata)

predictors <- as.matrix(traindata[,9:96])

response <- factor(traindata$Curated_Quality,
                   levels = c("good", "poor"),
                   labels = c("good", "poor"))

lM <- lossMatrix(c("good","good","poor","poor"),
                 c("good","poor","good","poor"),
                 c(     0,     1,     5,     0))

glmnetLRC_fit <- glmnetLRC(response, predictors, lossMat = lM,
                           estimateLoss = TRUE,
                           nJobs = parallel::detectCores())

save(glmnetLRC_fit, file = "../../data/glmnetLRC_fit.RData")
