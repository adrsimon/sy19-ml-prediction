library(mice)

initialize <- function(data_set, columns_to_remove, target_column, train_percentage=0.7) {
  set.seed(10)
  #suppression des colonnes à supprimer
  `%ni%` <- Negate(`%in%`)
  data_set <- subset(data_set,select = names(data_set) %ni% columns_to_remove)
  
  train_idx =  sample(nrow(data_set), floor(train_percentage * nrow(data_set)))
  X <- subset(data_set,select = names(data_set) %ni% c(target_column)) # retire y que temporairement et stocke dans X
  #X <- scale(X)
  train <- data_set[train_idx,]
  test <- data_set[-train_idx,]
  y <- data_set[,target_column]
  X_train <- X[train_idx, ] #attention , car matrice et pas vecteur
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  data_list = list(
    "X_train" = X_train,
    "y_train" = y_train,
    "X_test" = X_test,
    "y_test" = y_test,
    "train" = train,
    "test" = test
  )
  return(data_list) }

handle_missing_values <- function(data, method="pmm") {
  
  sprintf("Imputing missing data with the %s. method.", method)
  switch(method, 
         pmm = {
           
           #1.a: methode de base meth='pmm' de base dans mice() : fait référence à la méthode d'appariement moyen prédictif comme méthode d'imputation
           imputed_data <- mice(data, method='pmm', m=1) #5 imputations par variable
           completed_data <- complete(imputed_data, 1)
         },
         
         rf = {
           #2.b: methode random forest meth='rf'
           imputed_data <- mice(data, m=1, method = "rf")
           #data set complet 2:
           completed_data <- complete(imputed_data, 1)
         },
         
         stat_mean = {
           # la moyenne:
           # on calcule la moyenne pour chaque colonne en ignorant les données manquantes avec na.rm =  TRUE
           # les moyennes sont attribuées pour chaque colonne à chaque valeur manquante
           
           numerical_columns <- unlist(lapply(data, is.numeric), use.names = FALSE)
           completed_data <- data.frame(data[numerical_columns])
           for(i in 1:ncol(completed_data)){
             completed_data[is.na(completed_data[,i]), i] <- mean(completed_data[,i], na.rm = TRUE)
           }
           
         },
         
         stat_median = {
           # la median:
           # on calcule la médiane pour chaque colonne en ignorant les données manquantes avec na.rm =  TRUE
           # les moyennes sont attribuées pour chaque colonne à chaque valeur manquante
           
           numerical_columns <- unlist(lapply(data, is.numeric), use.names = FALSE)
           completed_data <- data.frame(data[numerical_columns])
           for(i in 1:ncol(completed_data)){
             completed_data[is.na(completed_data[,i]), i] <- median(completed_data[,i], na.rm = TRUE)
           }
           
         },
         
         delete = {
           #suppression descolonnes ayant des données manquantes 
           # completed_data <- data.frame(data)
           completed_data <- data[, colSums(is.na(data))==0]
         }
         
  )
  
  return(completed_data)
}


evaluate_models <- function(models, data_list) {
  model_names <- names(models)
  for (model_name in model_names) {
    cat("Evaluating model", model_name, "\n")
    model <- models[[model_name]]
    y_pred <- predict(model, as.matrix(data_list$X_test_new))
    mse = mean((data_list$y_test - y_pred)^2)
    rmse = sqrt(mse)
    cat("MSE=", mse, ", RMSE=", rmse, "\n")
    return (mse)
  } 
  return(mse)
  
}
### TESTING ZONE ###

# data <- read.csv(file = "data/communities_train.csv")
# sprintf("Data length: %s. .", length(data) )

# summary(data)
# summary(handled_data)


