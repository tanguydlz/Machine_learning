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

#Echantillonage des données
ech = sort(sample(nrow(datann), nrow(datann)*.7))

train = datann[ech, ]
test = datann[-ech, ]

#Definition variables explicatives/variable cible 
Xtrain = datann[, -17]
ytrain = datann[, 17]
Xtest = datann[, -17]
ytest = datann[, 17]

summary(train)

```


##Paramétrage du réseau de neurone
Nous allons paramétrer cette méthode grâce à de nombreux paramètres : 
 - hidden : spécifie le nombre de neuronnes dans la couche caché
 - err.fct : fonction utilisé pour determiner le calcul de l'erreur
 - rep : Nombre d'entrainement du réseau

 --------
La régularisation permet d'éviter le sur ou le sous ajustement des données d'entrainement.Elle consiste à modifier la fonction d'erreur du réseau en pénalisant les poids importants par l'ajout d'un terme d'erreur supplémentaire. Cependant si la constante de régularisation est trop importante cela entraine un sous ajustement, et si elle est trop peu importante alors cela entraine un sur-ajustement. Car cela augmente la réussite de généralisation. 
 --------

```{r}


# n <- neuralnet(ytrain~age+balance+duration, 
 #              data = train, 
  #             hidden = 5, 
   #            err.fct = "ce", 
    #           linear.output = FALSE, 
     #          lifesign = "full", 
      #         rep = 2, 
       #        algorithm = "rprop+", 
        #       stepmax = 100000)'''

nn=neuralnet(ytrain~., 
               data = train, hidden=10,act.fct = "logistic",
                linear.output = FALSE)

#hiden=3 : le reseau a 1 couche caché de 3 neurones

```

Cette fonction utilise d'autre paramètres comme : 
 - startweights : contient la valeurs de départ des poids (par défaut une initialisation aléatoire)
 
Représentation graphique du réseau de neurones séléctionné : 

```{r}
plot(nn)

```


##Prédiction de la variable cible y

Voici la distribution de la variable y prédit par le réseau de neurone :

```{r}
Predict=compute(nn,test)
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred

```

Voici la matrice de confusion : 
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






