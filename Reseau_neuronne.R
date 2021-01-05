---
title: "Data_mining"
author: "Amélie Picard"
date: "26/12/2020"
output: word_document
---


```{r}
#chargement des donnees
#setwd("C:/Users/ameli/Desktop/R/data_mining")

#le package neutralnet permet de réaliser des reseau de neurones :
library(neuralnet) 
library(dplyr)
library(glmnet)
library(caTools)

data<-read.csv2("bank.csv")

#Transformation de y en variable bianire : 
data$y = recode(data$y, "no" = 0, "yes" = 1)

```


#Réalisation du réseau de neurones

Nous allons réaliser un réseau de neurones à l'aide de la fonction "neutralnet" qui permet de réaliser tout type de réseau de neurones.

##Transformation des données

Avant de réaliser les réseaux de neurones, nous devons transformer les données. Pour les variables quantitatives, nous allons les normaliser pour qu'elle ai la même importance. Pour les variables qualitatives il faut les convertir en ordonné en leur donnant une valeur numeric de 1 à k ou k est le nombre de facteur. 
Avant de paramètrer le réseau de neuronne, il faut normaliser les données, les variables explicatives, dans notre cas toutes les variables du jeu de données sauf "y" la variables cible.

```{r}
#on récupère les données de base
datann<-data
dataquali<-data[,c(2,3,4,5,7,8,9,11,16)]

#quali en quali numeric
#fonction qui change pour une variable les noms des modalité par un nombre :
encoding<-function(var) {
    cpt=0
    for (i in levels(as.factor(var))) {
        for (j in length(var)) {
            var[var==i]<-cpt
        }
        cpt=cpt+1

    }
    return(as.numeric(var))
}
dataquali<-lapply(dataquali, encoding)

#donnée avec toute les variables numérique :
datann[,c(2,3,4,5,7,8,9,11,16)]<-dataquali

#normalisation des variables 
scaleddata<-scale(datann[-17])
datann<-data.frame(scaleddata,datann[17])
```

De plus, pour la vérification et la validation du meilleur reseau de neurone, nous découpons nos données en trois échantillons. Nous réalisons un premier découpage dans les données pour obtenir un échantillon d'apprentissage qui est constitué de 80% de la base et un échantillon test qui est constitué de 20% de la base. Puis nous re-découpons l'échantillon d'apprentissage en deux, pour récupérer un échantillon de validation qui est constitué de 30% de l'échantillon d'apprentissage. 

```{r}
#Echantillonage des données
ech = sort(sample(nrow(datann), nrow(datann)*.8))

train = datann[ech, ]
test = datann[-ech, ]

#Definition variables explicatives/variable cible de test
Xtest = test[, -17]
ytest = test[, 17]

#Découpage de l'échantillon d'apprentissage en ech de validation : 
#Echantillonage des données
ech2 = sort(sample(nrow(train), nrow(train)*.7))

valid = train[-ech2, ]
train = train[ech2, ]

#Definition variables explicatives/variable cible 
Xtrain = train[, -17]
ytrain = train[, 17]
Xvalid = valid[, -17]
yvalid = valid[, 17]

```


##Paramétrage du réseau de neurone
Nous allons paramétrer cette méthode grâce à de nombreux paramètres : 
 - hidden : spécifie le nombre de neuronnes dans la couche caché
 - err.fct : fonction utilisé pour determiner le calcul de l'erreur (ce : entropie croisée, sse : somme des erreurs au carré)
 - linear.output = FALSE : Ne pas réaliser de regression linéaire
 
 - rep : Nombre d'entrainement du réseau (pas utiliser pour éviter le sur-apprentissage)

Dans le cas d'une variable y catégorielle binaire, nous choisissons comme fonction d'erreur l'entropie croissée.

Premier modéle : 

```{r}
nn=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE)
attributes(nn)
nn$result.matrix

```

Cette fonction utilise d'autre paramètres comme : 
 - startweights : contient la valeurs de départ des poids (par défaut une initialisation aléatoire)

Validation du modèle : 

```{r}
Predict=compute(nn,valid)
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred

```

Voici la matrice de confusion entre les valeurs prédite et les valeurs test et le taux d'erreur :  
```{r}
m<-table(pred,yvalid)
m
#tx d'erreur :
tx<-(m[1,2])/length(yvalid)
tx
```

Deuxième modèle : 
```{r}
nn2=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE,likelihood = TRUE)

Predict2=compute(nn2,valid)
prob2 <- Predict2$net.result
pred2 <- ifelse(prob2>0.5, 1, 0)

m2<-table(pred2,yvalid)
tx2<-(m2[1,2])/length(yvalid)
tx2

```
Modéle 2 moins performant avec le likelihood (taux d'erreur = 8% > 7%)

Troixième modèle : augmentation du nombre de couche de la couche caché de 10 à 17 (nombre de variable d'entrée)
```{r}
nn3=neuralnet(ytrain~., data = Xtrain, hidden=17, err.fct="ce",linear.output = FALSE)

Predict3=compute(nn3,valid)
prob3 <- Predict3$net.result
pred3 <- ifelse(prob3>0.5, 1, 0)

m3<-table(pred3,yvalid)
tx3<-(m3[1,2])/length(yvalid)
tx3

```

Taux d'erreur un peu meilleur (7,3%<7,5%)

Quatrième modèle : augmentation du nombre de couche de la couche caché de 17 à 34 (deux fois le nombre de variable d'entrée)
```{r}
nn4=neuralnet(ytrain~., data = Xtrain, hidden=34, err.fct="ce",linear.output = FALSE)

Predict4=compute(nn4,valid)
prob4 <- Predict4$net.result
pred4 <- ifelse(prob4>0.5, 1, 0)

m4<-table(pred4,yvalid)
tx4<-(m4[1,2])/length(yvalid)
tx4

```

Taux d'erreur un peu élevé, donc moins intéressant (7,5%>7,3%)

Nous décidons de garder 17 neurones dans la couche caché :



##Prédiction de la variable cible y

Représentation graphique du réseau de neurones séléctionné : 

```{r}
plot(nn)

```

Voici la distribution de la variable y prédite par le réseau de neurone :

```{r}
Predict=compute(nn,test)
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred

```

Voici la matrice de confusion entre les valeurs prédite et les valeurs test : 
```{r}
m<-table(pred,ytest)
m
```

Voici le taux d'erreur de cette prédiction :
```{r}
#tx d'erreur :
tx<-(m[1,2])/length(ytest)
tx
```






