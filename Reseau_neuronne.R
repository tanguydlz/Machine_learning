---
title: "Data_mining"
author: "Amélie Picard"
date: "26/12/2020"
output: word_document
---


```{r}
#chargement des donnees
#setwd("C:/Users/ameli/Desktop/R/data_mining")

#le package nnet permet de réaliser dess reseau de neurones avec une seule couche caché :
library(nnet)

data<-read.csv2("bank.csv")

#Transformation de y en variable bianire : 
data$y = recode(data$y, "no" = 0, "yes" = 1)

#Echantillonage des données
ech = sort(sample(nrow(data), nrow(data)*.7))

train = data[ech, ]
test = data[-ech, ]

#Definition variables explicatives/variable cible 
Xtrain = train[, -17]
ytrain = train[, 17]
Xtest = test[, -17]
ytest = test[, 17]

summary(train)

```

#Paramétrage du réseau de neurones

Nous allons réaliser un réseau de neurones à l'aide de la fonction "nnet" qui prend en compte une couche caché. Nous allons tuner cette méthode grâce à de nombreux paramètre : 
 - size : spécifie le nombre de neuronnes dans la couche caché
 - decay : régularisation
 - maxit : Nombre maximum d'itération
 

```{r}

model <- nnet(y=ytrain, x=Xtrain, size = 10,)

par(mar=numeric(4),mfrow=c(1,2),family='serif')
plot(model,nid=F)
plot(model)

```

Cette fonction utilise d'autre paramètre comme : 
 - skip : Créé des liens entre la couche d'entrée et de sortie

#Prédiction de la variable cible y

```{r}
pred <- predict(model, newdata = test)
table(pred)
```

