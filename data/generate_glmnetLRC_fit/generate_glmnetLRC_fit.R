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

save(glmnetLRC_fit, file = "../glmnetLRC_fit.RData")
