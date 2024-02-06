# Récupération des données et analyse exploratoire

data <- read.table(file = 'data/phoneme_train.txt' , header = TRUE)

any(is.na(data)) # Aucune valeur manquante
nbcol <- ncol(data)
X <- data[, c(-nbcol)]
y <- data[, c(nbcol)]
y <- as.factor(y) 
data
summary(y) # On retrouve les classes et leurs effectifs dans l'échantillon
# L'échantillon semble contenir la moitié de la population totale avec une répartition similaire
barplot(table(y), xlab='Classes', ylab='Effectifs')

boxplot(X)
summary(X)
sapply(X,mean)
sapply(X,sd) # Les variables sont déjà centrées-réduites


# trace quelques instances, un échantillon, pour savoir de quoi on parle
matplot(1:256,t(X[1:15,]),col=y[1:15],type="l",
        xlab="Frequency",ylab="Log-periodogram")
legend("bottomright",legend=levels(y),lty=1,col=1:5)
# trace les moyennes pour chaque classe pour se rendre compte qu'on peut facilement différencier
# certaines, ce qui se retrouve dans les matrices de confusion plus tard
# peut faire ça sur un échantillon ou lisser pour le visuel
matplot(1:256, sapply(data[y=="aa",-c(257)],mean), type="l", col='green', lty=1, lwd=0.1, ylim=c(-4,4), xlab = "Frequency", ylab = "Log-periodogram")
matlines(sapply(data[y=="ao",-c(257)],mean), type = "l", col='cyan', lty = 1, lwd=0.1)
matlines(sapply(data[y=="dcl",-c(257)],mean), type = "l", col='magenta', lty = 1, lwd=0.1)
matlines(sapply(data[y=="iy",-c(257)],mean), type = "l", col='yellow', lty = 1, lwd=0.1)
matlines(sapply(data[y=="sh",-c(257)],mean), type = "l", col='red', lty = 1, lwd=0.1)

# test sur les 2 classes ao aa avec PCA
# test<-data[y=="aa",-c(257)]
# data<-rbind(test,data[y=="ao",-c(257)])


# Réduction de la dimension 
# Etape importante pour ce jeu de données car on a 256 variables ce qui est beaucoup
# On a même des erreurs pour certains modèles si on réduit pas la dimension (LR)
# Et des résultats catastrophiques pour d'autres (QDA)

# Les méthodes de réduction de la dimension incluant de la sélection de variables
# ne semble pas très pertinentes pour ce jeu de données car il s'agit de log-periodograms
# et que donc nos variables sont des fréquences, dont la sélection n'a pas beaucoup de sens

pca<-princomp(X)
lambda<-pca$sdev^2
Z<-pca$scores
plot(cumsum(lambda)/sum(lambda)*100,type="l",xlab="Dimensions",ylab="Variance expliquée (en %)")
# La PCA semble être une bonne option pour réduire la dimension étant donné 
# la forte variance expliquée par les premières composantes

# Représentation des données dans les premiers plans factoriels
pairs(Z[,c(1:3)],col=c("green","blue","red","cyan","orange")[y], upper.panel=NULL)
# dcl sh et iy se distingue facilement visuellement de tous les autres
# plus compliqué de distiguer ao et aa entre-eux, logique ce sont des sons proches

# En testant les différentes combinaisons des premiers plans factoriels,
# et d'après les matrices de confusion,
# il semble que les classes "ao" et "aa" sont proches et il est difficile de les distinguer, 
# pas les autres. 
# on peut étudier ces 2 classes séparement voir si un autre classifieur est plus 
# efficace uniquement sur ces classes et pas sur les autres
# à termes si c'est le cas, sur les données no label, il faudra
# séparer le travail, si classifieur est pas sûr de sa classif entre les 2 classes,
# il laisse l'autre classif plus performant sur ces données faire le travail
# classif en cascade

# On conserve les 50 premières composantes, bon compromis
# on déterminera le nb optimal de comp à prendre par CV sur les meilleurs modèles plus tard
data <- data.frame(cbind(Z[,c(1:55)],y))
data$y <- y

# Validation croisée pour différents modèles

n<-nrow(data)
K<-10

# Logistic Regression 

library(nnet)

folds=sample(1:K,n,replace=TRUE)
CV.LR<-rep(0,K)
for(k in (1:K)){
  lr<-multinom(y~.,data=data[folds!=k,])
  pred<-predict(lr,newdata=data[folds==k,])
  matrix.conf.lr <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.LR[k]<-(1-sum(diag(matrix.conf.lr))/n.test)
}
CV.LR

# LDA (truc de base)

library(MASS)

folds=sample(1:K,n,replace=TRUE)
CV.LDA<-rep(0,K)
for(k in (1:K)){
  lda<-lda(y~.,data=data[folds!=k,])
  pred<-predict(lda,newdata=data[folds==k,])
  matrix.conf.lda <- table(data$y[folds==k], pred$class)
  n.test<-nrow(data[folds==k,])
  CV.LDA[k]<-(1-sum(diag(matrix.conf.lda))/n.test)
}
CV.LDA
# avec dim=4
folds=sample(1:K,n,replace=TRUE)
CV.LDA.reduc<-rep(0,K)
for(k in (1:K)){
  lda<-lda(y~.,data=data[folds!=k,])
  pred<-predict(lda,newdata=data[folds==k,],dim=4)
  matrix.conf.lda <- table(data$y[folds==k], pred$class)
  n.test<-nrow(data[folds==k,])
  CV.LDA.reduc[k]<-(1-sum(diag(matrix.conf.lda))/n.test)
}
CV.LDA.reduc

# QDA

folds=sample(1:K,n,replace=TRUE)
CV.QDA<-rep(0,K)
for(k in (1:K)){
  qda<-qda(y~.,data=data[folds!=k,])
  pred<-predict(qda,newdata=data[folds==k,])
  matrix.conf.qda <- table(data$y[folds==k], pred$class)
  n.test<-nrow(data[folds==k,])
  CV.QDA[k]<-(1-sum(diag(matrix.conf.qda))/n.test)
}
CV.QDA

# Naive Bayes

library(naivebayes)

folds=sample(1:K,n,replace=TRUE)
CV.NB<-rep(0,K)
for(k in (1:K)){
  nb<-naive_bayes(y~.,data=data[folds!=k,])
  pred<-predict(nb,newdata=data[folds==k,])
  matrix.conf.nb <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.NB[k]<-(1-sum(diag(matrix.conf.nb))/n.test)
}
CV.NB

# KNN

library(FNN)

folds=sample(1:K,n,replace=TRUE)
folds.k=sample(1:K,n,replace=TRUE)
CV.KNN<-rep(0,K)

# On détermine le k optimal en effectuant une première validation croisée

# Fonction qui effectue une validation croisée pour k

err_k <- function(k) {
  CV.k<-rep(0,K)
  for (i in (1:K)){
    knn <- knn(data[folds.k!=i, -ncol(data)], data[folds.k==i, -ncol(data)], data$y[folds.k!=i], k)
    matrix.conf.KNN <- table(data$y[folds.k==i], knn)
    n.test<-nrow(data[folds.k==i,])
    CV.k[i]<-(1-sum(diag(matrix.conf.KNN))/n.test)
  }
  return(mean(CV.k))
}

errs <- sapply(1:25, err_k) # Applique pour k de 1 à 50
plot(1:25,errs)
kopt<-which.min(errs) # K optimal
kopt

# Puis on applique une validation croisée avec ce K opt

for(i in (1:K)){
  knn <- knn(data[folds!=i, -ncol(data)],data[folds==i, -ncol(data)], data$y[folds!=i], kopt) 
  matrix.conf.knn <- table(data$y[folds==i], knn)
  n.test<-nrow(data[folds==i,])
  CV.KNN[i]<-(1-sum(diag(matrix.conf.knn))/n.test)
}
CV.KNN

# MDA (Mixture => hypothèse que chaque classe est un mélange gaussien de sous-classes)

library(mda)

folds=sample(1:K,n,replace=TRUE)
CV.MDA<-rep(0,K)
for(k in (1:K)){
  mda<-mda(y~.,data=data[folds!=k,])
  pred<-predict(mda,newdata=data[folds==k,])
  matrix.conf.mda <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.MDA[k]<-(1-sum(diag(matrix.conf.mda))/n.test)
}
CV.MDA
# using generalized ridge penalty
folds=sample(1:K,n,replace=TRUE)
CV.MDA.ridge<-rep(0,K)
for(k in (1:K)){
  mda<-mda(y~.,data=data[folds!=k,],method=gen.ridge)
  pred<-predict(mda,newdata=data[folds==k,])
  matrix.conf.mda <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.MDA.ridge[k]<-(1-sum(diag(matrix.conf.mda))/n.test)
}
CV.MDA.ridge

# FDA (Flexible => utilise des combinaisons non linéaires des prédicteurs comme splines)

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
# using MARS method
folds=sample(1:K,n,replace=TRUE)
CV.FDA.mars<-rep(0,K)
for(k in (1:K)){
  fda<-fda(y~.,data=data[folds!=k,], method=mars)
  pred<-predict(fda,newdata=data[folds==k,])
  matrix.conf.fda <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.FDA.mars[k]<-(1-sum(diag(matrix.conf.fda))/n.test)
}
CV.FDA.mars
#using BRUTO method
folds=sample(1:K,n,replace=TRUE)
CV.FDA.bruto<-rep(0,K)
for(k in (1:K)){
  fda<-fda(y~.,data=data[folds!=k,],method=bruto)
  pred<-predict(fda,newdata=data[folds==k,])
  matrix.conf.fda <- table(data$y[folds==k], pred)
  n.test<-nrow(data[folds==k,])
  CV.FDA.bruto[k]<-(1-sum(diag(matrix.conf.fda))/n.test)
}
CV.FDA.bruto

# RDA (Regularized => comme en régression linéaire, compromis entre LDA et QDA avec lambda, C.4 - p.87)
# faut déterminer comment obtenir les paramètres optimaux, 2 CV imbriquées ou train de caret
# http://www.sthda.com/english/articles/36-classification-methods-essentials/146-discriminant-analysis-essentials-in-r/#mixture-discriminant-analysis---mda
# https://daviddalpiaz.github.io/r4sl/regularized-discriminant-analysis.html
# https://www.r-bloggers.com/2018/11/linear-quadratic-and-regularized-discriminant-analysis/
# https://topepo.github.io/caret/model-training-and-tuning.html
# https://rdrr.io/cran/caret/man/models.html

# RDA est tr�s robuste face aux fortes multicolin�arit�s dans les donn�es comme dans notre dataset
# de log-periodogram, $posterior � tester

library(klaR)
library(caret)

folds=sample(1:K,n,replace=TRUE)
CV.RDA<-rep(0,K)
rdaGrid <-  expand.grid(gamma = 0.16, lambda = 0.95)
for(k in (1:K)){
  rda <- train(y~.,data=data[folds!=k,],method="rda",tuneGrid=rdaGrid)
  #rda<-train(y~.,data=data[folds!=k,], "rda", trControl = trainControl(method = "cv"))
  #rda<-rda(y~.,data=data[folds!=k,])
  pred<-predict(rda,newdata=data[folds==k,])
  CV.RDA[k]<-1-mean(pred == data$y[folds==k])
}
CV.RDA

# Affichage des résultats 
boxplot(CV.LR, CV.LDA, CV.LDA.reduc, CV.QDA, CV.NB, CV.KNN, CV.MDA, CV.MDA.ridge, CV.FDA, CV.FDA.mars, CV.FDA.bruto, CV.RDA, names=c("LR", "LDA", "LDA 4D", "QDA", "NB", "KNN", "MDA", "MDA Ridge", "FDA", "FDA Mars", "FDA Bruto", "RDA"), main="Taux d'erreur sur une validation croisée à 10 plis")

# Combien de composantes de la PCA ?
folds.k=sample(1:K,n,replace=TRUE)

err_k <- function(i) {
  CV.k<-rep(0,K)
  data <- data.frame(cbind(Z[,c(1:i)],y))
  for(k in (1:K)){
    lda<-lda(y~.,data=data[folds!=k,])
    pred<-predict(lda,newdata=data[folds==k,])
    CV.k[k]<-1-mean(pred$class == data$y[folds==k])
  }
  return(mean(CV.k))
}

errs <- sapply(1:256, err_k) # Applique pour k de 1 à 50
plot(1:256,errs)
kopt<-which.min(errs) # K optimal
kopt #55

# Combien de composantes de LDA ?
folds.k=sample(1:K,n,replace=TRUE)
err_k <- function(i) {
  CV.k<-rep(0,K)
  for(k in (1:K)){
    lda<-lda(y~.,data=data[folds!=k,])
    pred<-predict(lda,newdata=data[folds==k,], dim=i)
    CV.k[k]<-1-mean(pred$class == data$y[folds==k])
  }
  return(mean(CV.k))
}

errs <- sapply(1:20, err_k) # Applique pour k de 1 à 50
plot(1:20,errs)
kopt<-which.min(errs) # K optimal
kopt #4

# https://www.datascienceblog.net/post/machine-learning/linear-discriminant-analysis/
# http://statweb.lsu.edu/faculty/li/teach/exst7152/phoneme-example.html
# pur banger, travail avec le même dataset

# SVM et KPCA
# https://rpubs.com/uky994/593668