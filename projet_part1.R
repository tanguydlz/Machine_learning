#setwd("C:/Users/Axelle/Desktop/M/03_SISE/05_MACHINE LEARNING/Projet")

#Lecture et description des données
#2
D = read.table("breast-cancer-wisconsin.data", sep = ",", na.strings = "?")
colnames(D) = c("code_number", "clump_thickness", "cell_size_uni", "cell_shape_uni", "marginal_adhesion", "single_epithetial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class")

#3
class(D) #df
str(D) #nbr obs/var et type de chaque var
head(D) #les premières lignes
summary(D) #résumé de chaque var

#Separation des donnees en “train” et “test”

#4
na_row = complete.cases(D)[complete.cases(D)==FALSE]  
length(na_row) #on a bien 16 lignes non completes

#5
D = D[complete.cases(D),]
nrow(D) #on a bien 699-16 = 683 lignes

#6
X = D[, 2:10] #données explicatives
y = D$class #variable cible

#7
library(dplyr)
length(y[y==2]) #444
length(y[y==4]) #239
y = recode(y, "2" = 0, "4" = 1)
length(y[y==0]) #444 -> benin
length(y[y==1]) #239 -> maligne
#on a bien recodé

#8
benin = which(y == 0, arr.ind = TRUE)
#length(benin)

#9
#train_set = benin[1:200]
Xtrain_set = X[benin[1:200],]
Xtest_set = X[-benin[1:200],]
ytrain_set = y[benin[1:200]]
ytest_set = y[-benin[1:200]]


#One-class SVM
#10
library(e1071)

#11
oc_svm_fit = svm(ytrain_set ~ ., data=Xtrain_set, type = "one-classification", gamma = 1/2)

#12
oc_svm_pred_test = predict(oc_svm_fit, newdata = Xtest_set, decision.values = TRUE)
oc_svm_pred_test

#13
attr(oc_svm_pred_test, "decision.values") #les scores des observations
oc_svm_score_test = -as.numeric(attr(oc_svm_pred_test ,"decision.values")) #on a changer le signe (pk?)

#Courbe ROC
#14
library(ROCR)

#15
pred_oc_svm = prediction(oc_svm_score_test, ytest_set)
oc_svm_roc = performance(pred_oc_svm, measure = "tpr", x.measure = "fpr")
plot(oc_svm_roc)

#16
#TP = sensibilite
#FP = 1-specificité
#la courbe ROC est au dessus du classifieur aleatoire -> bonne classification

oc_svm_auc <- performance(pred_oc_svm, "auc")
oc_svm_auc@y.values[[1]]
#0.99 --> tres bonne performance

#Kernel PCA
library(kernlab)
kernel = rbfdot(sigma = 1/8) #kernel generating function
Ktrain = kernelMatrix(kernel, as.matrix(Xtrain_set))

#18
#question: c'est quoi n? le nombre de ligne de Ktrain? oui
k2 = apply(Ktrain, 1, sum)
k3 = apply(Ktrain, 2, sum)
k4 = sum (Ktrain)
n = ncol(Ktrain) #n = nbr de ligne de Xtrain_set = nombre de données de l'ensemble d'apprentissage
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



