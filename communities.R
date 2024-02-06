### IMPORTATION ###

library(VIM)
library(FactoMineR)
library(missMDA)
library(ggplot2)
library(naniar)
library(tidyr)
library(stringr)
library(mice)
library(Hmisc)

library(glmnet)
library(dplyr)
library(rpart)
library(data.table)
library(caret)
library(xgboost)
source("communities_utils.R")

data = read.csv(file = "data/communities_train.csv")
target_column = "ViolentCrimesPerPop" # var de réponse

### 1 - EDA sur les données ###
#------------------------------------------------------------------------------

#faire une validation croisée 
plot(data[target_column])
str(data) # pour avoir une meilleure vue des données
any(is.na(data)) # renvoie True car il ya des données manquantes
colSums(is.na(data)) # affichage de chaque colonne avec son nombre de valeurs NA
dim(na.omit(data)) # nombre total de valeurs NA dans tout le dataset : 59 128

gg_miss_var(data, show_pct = TRUE)

#on peut avoir la liste des noms des colonnes ayant des na values: on a 25 variables avec des valeurs na
list_all <- colnames(data)
length(list_all)
list_na <- colnames(data)[ apply(data, 2, anyNA) ] #argument 2 pour appliquer apply sur  les colonnes
length(list_na)
cat(list_na,sep=", ")
md.pattern(data) # pour voir combien d'échantillons sont complets et combien d'échantillons manquent telle valeur sur l'ensemble des données
#plot pour expliquer le nombre de variables manquantes: semble ne pas être aléatoire

aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

#Représentation des valeurs manquantes :




### 2 - Gestion des NaN ###
#------------------------------------------------------------------------------

# 4 methods are available : pmm, rf, stat_mean, stat_median, delete

handled_data <- handle_missing_values(data, method = "delete")

### 3 - Initialisation des données
#------------------------------------------------------------------------------

data_list <- initialize(handled_data, c("fold", "communityname", "state"), target_column) # à enlever : indiqué dans communities.name

str(data_list)
any(is.na(data_list))
### 4 - Sélection/Réduction de la dimension
#------------------------------------------------------------------------------

# RFE (recursive feature elimination) sur le nouveau data_set après pré-traitement
# A FAIRE : fonctionnaliser cette partie pour sélectionner la méthode

ctrl <- rfeControl(functions = lmFuncs, #optimiser fonction suivante
                   method = "repeatedcv", # prend morceaux dataset un par un crossvalidation
                   repeats = 5,
                   verbose = FALSE)


lmProfile <- rfe(data_list$X_train, data_list$y_train,
                 rfeControl = ctrl,
                 metric = "Rsquared",
                 sizes=seq(1, 100, by=5))

new_features = predictors(lmProfile)
print(new_features)
length(new_features)
data_list$X_train_new = data_list$X_train[,new_features]
data_list$X_test_new = data_list$X_test[,new_features]
data_list$train_new = data_list$train[,c(new_features, target_column)]
data_list$test_new = data_list$test[,c(new_features, target_column)]

ggplot(data = lmProfile, metric = "RMSE") + theme_bw() # la plus faible
ggplot(data = lmProfile, metric = "Rsquared") + theme_bw() #plus grande




pca<-princomp(data_list$X_train) #delete : 26 
lambda<-pca$sdev^2
lambda
Z<-pca$scores
plot(cumsum(lambda)/sum(lambda)*100,type="l",xlab="Dimensions",ylab="Variance explique (en %)")


# Print the summary
print(summary(pca))

# Determine the number of features that explain 95% of the variance
num_features <- sum(pca$sdev >= 0.70)

print(paste("Number of features:", num_features))


### 5 - Model building
#------------------------------------------------------------------------------

#Methodes

#Ridge regression caret
ridge_1 <- train(data_list$X_train_new, data_list$y_train, method = "glmnet",
               trControl=trainControl(method = "cv"),
               preProcess = c("center", "scale"),
               tuneGrid = expand.grid(alpha = 0.1,
                                      lambda = 0), metric="Rsquared"
)

ridge_caret

#ridge normale:
ridge_2 <- cv.glmnet(as.matrix(data_list$X_train_new), as.matrix(data_list$y_train), alpha = 0, type.measure = "mse", family="gaussian")

#Lasso regression:
lasso_2 <- cv.glmnet(as.matrix(data_list$X_train_new), as.matrix(data_list$y_train), type.measure="mse", alpha=1, family="gaussian")

#lasso regression caret:

lasso_1 <- train(data_list$X_train_new, data_list$y_train, method = "lasso", trControl = trainControl(method = "cv", number = 5), tuneGrid = expand.grid(.fraction = 0.6),  metric="Rsquared")
lasso_1

#knn simple:
knn_reg <- knnreg(x=data_list$X_train_new, y=data_list$y_train, nfolds = 4)

#knn caret


knn_2 <- train(x=data_list$X_train_new, y=data_list$y_train, method = "knn", metric="RMSE", tuneGrid = expand.grid(k = 6), trControl = trainControl(method="cv", number = 5))
knn_2

#nn caret
library(qrnn)
nn_2 <-  train(x=data_list$X_train_new, y=data_list$y_train, method = "nnet", metric="RMSE", tuneGrid = expand.grid(size=10, decay=0.1), trControl = trainControl(method="cv", number = 5))
nn_2
nn <- train(x=data_list$X_train_new, y=data_list$y_train, method = "qrnn", metric="RMSE", tuneGrid = expand.grid(n.hidden=2, penalty=0.1, bag= "TRUE"), trControl = trainControl(method="cv", number = 5))


#reseaux  de neurones
'library(tabnet)
library(recipes)
library(yardstick)
library(torch)
install_torch()
rec <- recipe(ViolentCrimesPerPop ~ ., data = data_list$train) %>% 
  step_normalize(all_numeric(), -all_outcomes())

fit <- tabnet_fit(rec, data_list$train, epochs = 40, loss = "mse")
cbind(data_list$test, predict(fit, data_list$test)) %>% 
  rmse(ViolentCrimesPerPop, estimate = .pred)

predict_dou <- as.double(predict(fit, data_list$X_test))
result <- rmse_vec(data_list$y_test, predict(fit, data_list$X_test))

typeof(predict(fit, data_list$X_test))
suppressWarnings(autoplot(fit))'
#SVM basic: 
library(e1071)
svm = svm(x=data_list$X_train_new, y=data_list$y_train,kernel = "linear", cost = 14, scale = FALSE)

#SVM caret:
ctrl <- trainControl(method="cv",
                     number = 2,
                     summaryFunction=twoClassSummary,
                     classProbs=TRUE)

# Grid search to fine tune SVM
#grid <- expand.grid(.sigma=c(0.001, 0.01, 0.1), .C=c(10,100,1000))


svmreg <- train(x=data_list$X_train_new, y=data_list$y_train,method='svmRadialSigma', tuneGrid=expand.grid(sigma=0.01, C=9), trControl = trainControl(method="cv", number = 5))

# MEILLEURE MSE : 0.01785634
svmreg
#gbm base 
library(gbm)
#btree = gbm(data_list$X_train_new, data_list$y_train, distribution = "bernoulli", bag.fraction = 0.5, n.trees = 1000, interaction.depth =6, shrinkage = 0.1, n.minobsinnode = 10)

# Random forest gbm
#Boosting par gradient : GBT construit des arbres un par un, où chaque nouvel arbre aide à corriger les erreurs commises par l'arbre précédemment formé

boosted_tree <- train(x=data_list$X_train_new, y=data_list$y_train, method = "gbm",  tuneGrid=expand.grid(interaction.depth = 5, n.trees = 1250, shrinkage = 0.005, n.minobsinnode = 10),trControl = trainControl(method="cv", number = 5), verbose=TRUE)

# Random forest rf
#Les RF forment chaque arbre indépendamment, en utilisant un échantillon aléatoire des données: contribue à rendre le modèle plus robuste qu'un arbre de décision unique, et moins susceptible d'être surajusté aux données d'apprentissage. 


random_forest <- train(x=data_list$X_train_new, y=data_list$y_train, method = "ranger", tuneGrid=expand.grid(mtry = 10, splitrule = "extratrees" , min.node.size = 1), num.trees=1000, trControl = trainControl(method="cv", number = 5), verbose=TRUE)
#RMSE: 0.138 MSE: 0.0189883 


#xg boost 
dtrain <- xgb.DMatrix(label = data_list$y_train, data = as.matrix(data_list$X_train_new))

xg_boost <- xgboost(data = dtrain, max.depth = 4,
               eta = 0.01, subsample=0.5, nround = 3000, print_every_n = 250, objective = "reg:squaredlogerror")

importance_matrix = xgb.importance(colnames(data_list$X_train_new), model = bst)
importance_matrix
xgb.plot.importance(importance_matrix)


#affichage erreur chaque modele validation croisée : 

evaluate_models <- function(models, data_list) {
  model_names <- names(models)
  for (model_name in model_names) {
    cat("Evaluating model", model_name, "\n")
    model <- models[[model_name]]
    y_pred <- predict(model, as.matrix(data_list$X_test_new))
    mse = mean((data_list$y_test - y_pred)^2)
    rmse = sqrt(mse)
    cat("MSE=", mse, ", RMSE=", rmse, "\n")

  } 
}


models <- list(ridge_1=ridge_1, knn=knn_reg,lasso_2 = lasso_2, ridge_2=ridge_2, boosted_tree=boosted_tree, xg_boost=xg_boost, knn_2 = knn_2, lasso_1 = lasso_1, svm_simple = svm, svm_reg = svmreg, random_forest = random_forest, nn = nn, nn_2 = nn_2)
evaluate_models(models, data_list)
 
output <- capture.output(evaluate_models(models, data_list))
output
output_df = read.table(text=output, header = FALSE, sep = ",")
output_df

View(output_df)
ggplot(x, aes(x=model, y=performance)) + geom_boxplot() + labs(x="Model", y="Performance")

model = c("ridge_1", "knn_reg","lasso_2", "ridge_2", "boosted_tree", "xg_boost", "knn_2", "lasso_1", "svm", "svmreg", "random_forest", "nn", "nn_2")
mse = c(0.01956,0.02346146, 0.01958072, 0.02020578, 0.01834539, 0.01888775, 0.02252907, 0.01930047, 0.02067142, 0.01785634, 0.018805410, 0.02309449, 0.01817031 )


results <- data.frame(model=model, mse=mse)

ggplot(results, aes(x=model, y=mse)) +
  geom_boxplot() +
  labs(x="Model", y="MSE")


#affichage cross validation box plot sur valeurs finales :
results <- lapply(models, function(x) {
  train(x=data_list$X_train_new, y=data_list$y_train, method = models, metric = "MSE", trControl = trainControl(method = "cv"))
})


lapply(results, function(x) x$results)


scores <- resamples(boosted_tree, data_list, target_column, metric="MSE", trControl=trainControl(method="cv", number=5))

mean_score <- mean(scores$results$Accuracy)

print(paste("Mean score:", mean_score))


folds=sample(1:K,n,replace=TRUE)
CV.FDA<-rep(0,K)
for(k in (1:K)){
  fda<-fda(y~.,data=data[folds!=k,])
  pred<-predict(fda,newdata=data[folds==k,])
  matrix.conf.fda <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.FDA[k]<-(1-sum(diag(matrix.conf.fda))/n.test)
}
CV.FDA


