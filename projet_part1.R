#setwd("C:/Users/Axelle/Desktop/M/03_SISE/05_MACHINE LEARNING/Projet")
#setwd("C:/Users/ameli/Desktop/R/data_mining")

#Lecture et description des donnees
#2
D = read.table("breast-cancer-wisconsin.data", sep = ",", na.strings = "?")
colnames(D) = c("code_number", "clump_thickness", "cell_size_uni", "cell_shape_uni", "marginal_adhesion", "single_epithetial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class")

#3
class(D) #df
str(D) #nbr obs/var et type de chaque var
head(D) #les premieres lignes
summary(D) #resume de chaque var

#Separation des donnees en 'train' et 'test'

#4 
na_row = complete.cases(D)[complete.cases(D)==FALSE]  
length(na_row) #on a bien 16 lignes non completes

#5 Garder les donnees complete
D = D[complete.cases(D),]
nrow(D) #on a bien 699-16 = 683 lignes

#6 Création des variable X et Y :
X = D[, 2:10] #donnees explicatives
y = D$class #variable cible

#7 Recodage de Y :
library(dplyr)
#Distribution de y
length(y[y==2]) #444
length(y[y==4]) #239
y = recode(y, "2" = 0, "4" = 1)
#Vérification de la distribution de y aprés recodage :
length(y[y==0]) #444 -> benin
length(y[y==1]) #239 -> maligne
#on a bien recode

#8 Découpage des variables benin et malin
benin = which(y == 0, arr.ind = TRUE)
#length(benin)
malin = which(y == 1, arr.ind = TRUE)
#length(malin)


#9 Selection des 200 observations bégnines
Xtrain_set = X[benin[1:200],]
Xtest_set = X[-benin[1:200],]
ytrain_set = y[benin[1:200]]
ytest_set = y[-benin[1:200]]


#One-class SVM
#10 Chargement de la library
library(e1071)

#11 Estimation du modele avec noyau gaussien et de type "one-classification"
oc_svm_fit = svm(ytrain_set ~ ., data=Xtrain_set, type = "one-classification", gamma = 1/2)

#12 Score des observations de test :
oc_svm_pred_test = predict(oc_svm_fit, newdata = Xtest_set, decision.values = TRUE)
oc_svm_pred_test

#13
attr(oc_svm_pred_test, "decision.values") #les scores des observations
oc_svm_score_test = -as.numeric(attr(oc_svm_pred_test ,"decision.values")) #on a changer le signe (pk?)

#Courbe ROC
#14 Chargement library
library(ROCR)

#15 
pred_oc_svm = prediction(oc_svm_score_test, ytest_set)
oc_svm_roc = performance(pred_oc_svm, measure = "tpr", x.measure = "fpr")
plot(oc_svm_roc)

#16 
#TP = sensibilite
#FP = 1-specificite
#la courbe ROC est au dessus du classifieur aleatoire -> bonne classification
#Le modéle est performant car le point le plus élevé est trés proche du point(0,1) 
#qui est le point idéal

oc_svm_auc <- performance(pred_oc_svm, "auc")
oc_svm_auc@y.values[[1]]
#L'air sour la courbe est de 0.99, ce qui est très proche de la meilleur
#valeur qui est 1.

#6 Kernel PCA
library(kernlab)
kernel = rbfdot(sigma = 1/8) #kernel generating function
#echatillon d'entrainement :
Ktrain = kernelMatrix(kernel, as.matrix(Xtrain_set))

#18
#question: c'est quoi n? le nombre de ligne de Ktrain? oui
k2 = apply(Ktrain, 1, sum)
k3 = apply(Ktrain, 2, sum)
k4 = sum (Ktrain)
n = ncol(Ktrain) #n = nbr de ligne de Xtrain_set = nombre de donnees de l'ensemble d'apprentissage
KtrainCent = matrix (0, ncol = n, nrow = n) 
for(i in 1:n){
  for(j in 1:n){
    KtrainCent[i, j] = Ktrain[i, j] - 1/n*k2[i] - 1/n*k3[j] + 1/n^2*k4
  }
}
#k2 -> 1eme somme
#k3 -> 2eme somme
#k4 -> 3eme somme (la double somme)

#19
eigen_KtrainCent = eigen(KtrainCent)

#20
s = 80
A = eigen_KtrainCent$vectors[, 1:s]%*%diag(1/sqrt(eigen_KtrainCent$values[1:s]))

#21
K = kernelMatrix(kernel, as.matrix(X))

#22



