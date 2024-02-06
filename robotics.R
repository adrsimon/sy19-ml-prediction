library(MASS)
library(caret)
library(randomForest)

data <- read.table('data/robotics_train.txt')

boxplot(data)
plot(data)

n <- nrow(data)
m <- ncol(data)

cor(data)

train_index <- sample(n, round(2 * n / 3))
data_train <- data[train_index, ]
data_test <- data[-train_index, ]


plot(data_test$y, col=1)

ridge_model <- train(
  y ~ ., 
  data=data_train, 
  method="glmnet",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  tuneGrid = expand.grid(alpha = 0, lambda = 1)
)
ridge_pred <- predict(ridge_model, newdata=data_test, type="raw")
ridge_mean <- mean((data_test$y - ridge_pred)^2)
points(ridge_pred, col=2)

lasso_model <- train(
  y ~ ., 
  data=data_train, 
  method="glmnet",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  tuneGrid = expand.grid(alpha = 1, lambda = 1)
)
lasso_pred <- predict(lasso_model, newdata=data_test, type="raw")
lasso_mean <- mean((data_test$y - lasso_pred)^2)
sqrt(lasso_mean)

plot(lasso_pred)


pca <- princomp(data)
pca$scores
knn_model <- train(
  y ~ ., 
  data=data_train, 
  method="knn",
  metric="RMSE",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  preProcess = c('center','scale'),
  tuneGrid = expand.grid(k = 8)
)
knn_pred <- predict(knn_model, newdata=data_test, type="raw")
knn_mean <- mean((data_test$y - knn_pred)^2)
sqrt(knn_mean)
?knn
points(knn_pred, col=3)

kknn_model <- train(
  y ~ ., 
  data=data_train, 
  method="kknn",
  metric="RMSE",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  preProcess = c('center','scale'),
  tuneGrid = expand.grid(kmax=8, distance=2, kernel = "gaussian")
)

kknn_pred <- predict(kknn_model, newdata=data_test, type="raw")
kknn_mean <- mean((data_test$y - kknn_pred)^2)
sqrt(kknn_mean)



??train

gam_model <- train(
  y ~ ., 
  data=data_train, 
  method="gam",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  preProcess = c('center','scale'),
  tuneGrid = expand.grid(k = c(1, 5, 10))
)
gam_pred <- predict(gam_model, newdata=data_test, type="raw")
gam_mean <- mean((data_test$y - gam_pred)^2)
points(gam_pred, col=4)
??train

varImp(knn_model)
varImp(model)


library(leaps)

model <- regsubsets(y ~ ., data = data_train, method = "forward")
best_model <- summary(model)$which[which.max(summary(model)$adjr2)]
selected_predictors <- names(data_train)[best_model]


plot(data$y, data$X4)



library(neuralnet)
nn <- neuralnet(y ~ ., data = data_train, hidden=c(5, 5))
predictions <- compute(nn, newdata = data_test)
plot(nn)
plot(data$y, data$X4)


nn <- train(
  y ~ .,
  data=data_train,
  method='nnet',
  tuneGrid=expand.grid(size=c(99),decay=c(0.01)),
  preProcess = c('center','scale')
)

prednn <- predict(
  nn,
  newdata=data_test
)

mean((prednn - data_test$y)^2)

# 90 / 0.005 / 0.01113
# 99 / 0.010 / 0.01059
# 90 / 0.015 / 0.01098
# 99 / 0.015 / 0.01049
# 99 / 0.020 / 0.01042
# 99 / 0.025 / 0.01044
# 99 / 0.025 / 0.01174 / CV *03
# 99 / 0.025 / 0.01043 / CV *10
# 99 / 0.050 / 0.01142

varImp(nn)






library(mgcv)

data <- read.table('data/robotics_train.txt')

n <- nrow(data)

train_index <- sample(n, round(2 * n / 3))
data_train <- data[train_index, ]
data_test <- data[-train_index, ]

data_train = as.data.frame(data_train)

model <- gam(
  y ~ s(X6) + s(X5) + te(X4) + s(X7) + s(X2),
  data = data_train
)

predictions <- predict(
  model,
  newdata = data_test,
  type="response",
  se = TRUE
)


mean((data_test$y - predictions$fit)^2)


summary(model)
varImp(model)
plot(model)

AIC(model)
summary(model)$r.sq





data_ordered = data[order(data$y),]


train_index <- sample(n, round(2 * n / 3))
data_train <- data[train_index, ]
data_test <- data[-train_index, ]
knn_model <- train(
  exp(y) ~ ., 
  data=data_train, 
  method="knn",
  metric="RMSE",
  trControl=trainControl(
    method="repeatedcv",
    number=10,
    repeats=3
  ),
  preProcess = c('center','scale'),
  tuneGrid = expand.grid(k = 8)
)
knn_pred <- predict(knn_model, newdata=data_test, type="raw")
knn_mean <- mean((data_test$y - log(knn_pred))^2)








library(mgcv)

reggam <- gam(y ~ 
                s(X4, bs='ps') + 
                s(X2) + 
                X8 + 
                s(X6) + 
                s(X1) + 
                X3 + 
                s(X5) + 
                s(X7), 
              method='REML',
              data = data_train
              )

summary(reggam)
plot(reggam$linear.predictors, data_train$y)
predgam <- predict(reggam, newdata=data_test)
mean((predgam - data_test$y)^2)
varImp(reggam)
plot(reggam)

predictors <- data[, -9]



gamreg <- train(y ~ .,
                data = data_train,
                method="gam",
                trControl = trainControl(method="repeatedcv", number=10, repeats=1, verboseIter = TRUE),
                preProcess = c('center','scale')
)

sigmas <- c(0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195)
Cs <- c(3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7)

library(caret)

svmreg <- train(y ~ ., 
                data = data,
                method='svmRadialSigma',
                tuneGrid=expand.grid(sigma=c(0.163), C=c(4.7)),
                trControl = trainControl(method="repeatedcv", number=5, repeats=1, verboseIter = TRUE),
                preProcess = c('center','scale')
                )
svmpred <- predict(svmreg, newdata=data_test)
mean((svmpred - data_test$y)^2)

library(tidyverse)
ggplot(data=as.data.frame(svmreg$results), aes(x=svmreg$results$C, y=svmreg$results$sigma)) + geom_raster(aes(fill = svmreg$results$RMSE^2))


# 0.160 / 05. / 0.00686
# 0.150 / 10. / 0.00673
# 0.160 / 08. / 0.00673
# 0.165 / 07. / 0.00657
# 0.168 / 6.5 / 0.00669
# 0.165 / 6.3 / 0.00685
# 0.165 / 6.0 / 0.00679
# 0.174 / 4.0 / 0.00689
#163/4.7/0.0061
#162/4.5/0.00613



library(caret)

svmreg <- train(y ~ ., 
                data = data_train,
                method='svmRadialCost',
                tuneGrid = expand.grid(C=c(50, 80)),
                trControl = trainControl(method="repeatedcv", number=10, repeats=1, verboseIter = TRUE),
                preProcess = c('center','scale')
)

# 50 / 0.0074


svmreg <- train(y ~ ., 
                data = data,
                method='xgbLinear',
                trControl = trainControl(method="repeatedcv", number=2, repeats=1, verboseIter = TRUE),
                preProcess = c('center','scale')
)



#cor(data)

library(car)
vifs <- vif(predictors)
cor(data)
plot(cor(data))
rcorr(as.matrix(data))
library(Hmisc)
plot(data$X6, data$X8)
