#DATA 1:
library(caret)
library(kernlab)

data <- read.table(file = 'data/phoneme_train.txt' , header = TRUE)
nbcol <- ncol(data)
X <- data[, c(-nbcol)]
y <- data[, c(nbcol)]
y <- as.factor(y)
colnames(X)<-c(1:256)
pca <- prcomp(X, scale=T)
Z <- pca$x
data <- data.frame(cbind(Z[,c(1:55)],y)) 
data$y <- y

model.phoneme <- ksvm(as.factor(y)~.,data=data, type="C-svc",kernel="vanilladot",C=0.07)

prediction_phoneme <- function(dataset){
  library(caret)
  library(kernlab)
  colnames(dataset)<-c(1:256)
  Z <- predict(pca, dataset)
  dataset <- data.frame(Z[,c(1:55)])
  return(predict(model.phoneme,dataset))$class
}
data <- read.table(file = 'data/phoneme_train.txt' , header = TRUE)
mean(prediction_phoneme(data[,c(-ncol(data))])==data$y)

#DATA 2:
library(caret)
data_robotics <- read.table('data/robotics_train.txt')

model.robotics <- train(y ~ ., 
                        data = data_robotics,
                        method='svmRadialSigma',
                        tuneGrid=expand.grid(sigma=c(0.162), C=c(4.5)),
                        trControl = trainControl(method="repeatedcv", number=5, repeats=1, verboseIter = TRUE),
                        preProcess = c('center','scale')
)

prediction_robotics <- function(test_data) {
  library(caret)
  pred <- predict(
    model.robotics,
    newdata=test_data
  )
  return(pred)
}


#DATA 3:
library(caret)
source("communities_utils.R")

data_communities = read.csv(file = "data/communities_train.csv")
target_column = "ViolentCrimesPerPop"

handled_data <- handle_missing_values(data_communities, method = "delete")



data_list <- initialize(handled_data, c("fold", "communityname", "state"), target_column, train_percentage = 0.99)


ctrl <- rfeControl(functions = lmFuncs, #optimiser fonction suivante
                   method = "repeatedcv", # prend morceaux dataset un par un crossvalidation
                   repeats = 5,
                   verbose = FALSE)


lmProfile <- rfe(data_list$X_train, data_list$y_train,
                 rfeControl = ctrl,
                 metric = "Rsquared",
                 sizes=seq(1, 100, by=5))

new_features = predictors(lmProfile)
length(new_features)
print(new_features)
data_list$X_train_new = data_list$X_train[,new_features]
data_list$X_test_new = data_list$X_test[,new_features]
data_list$train_new = data_list$train[,c(new_features, target_column)]
data_list$test_new = data_list$test[,c(new_features, target_column)]

model.communities <- svmreg <- train(x=data_list$X_train_new, y=data_list$y_train, method='svmRadialSigma',tuneGrid=expand.grid(sigma=0.01, C=9), trControl = trainControl(method="cv", number = 5))



prediction_communities <- function(test_set) {
  library(caret)
  library(MASS)
  library(ranger)
  test_set <- test_set[,new_features]
  test_set <- handle_missing_values(test_set, method = "delete") # il faut refaire delete car faite sur l'entraineemnt mais pas sur nouvelles données 
  y_pred <- predict(model.communities, test_set)
  print(y_pred)
}

prediction_communities(data_communities)


save(
  "pca",
  "handle_missing_values",
  "model.phoneme",
  "prediction_phoneme",
  "model.robotics",
  "prediction_robotics",
  "new_features",
  "model.communities",
  "prediction_communities",
  file = "env.Rdata"
)
