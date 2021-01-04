---
title: "Data_mining"
author: "Am�lie Picard"
date: "26/12/2020"
output: word_document
---


```{r}
#chargement des donnees
#setwd("C:/Users/ameli/Desktop/R/data_mining")

#le package neutralnet permet de r�aliser des reseau de neurones :
library(neuralnet) 
library(dplyr)
library(glmnet)
library(caTools)

data<-read.csv2("bank.csv")

#Transformation de y en variable bianire : 
data$y = recode(data$y, "no" = 0, "yes" = 1)

```


#R�alisation du r�seau de neurones

Nous allons r�aliser un r�seau de neurones � l'aide de la fonction "neutralnet" qui permet de r�aliser tout type de r�seau de neurones.

##Transformation des donn�es

Avant de r�aliser les r�seaux de neurones, nous devons transformer les donn�es. Pour les variables quantitatives, nous allons les normaliser pour qu'elle ai la m�me importance. Pour les variables qualitatives il faut les convertir en ordonn� en leur donnant une valeur numeric de 1 � k ou k est le nombre de facteur. 
Avant de param�trer le r�seau de neuronne, il faut normaliser les donn�es, les variables explicatives, dans notre cas toutes les variables du jeu de donn�es sauf "y" la variables cible.

```{r}
#on r�cup�re les donn�es de base
datann<-data
dataquali<-data[,c(2,3,4,5,7,8,9,11,16)]

#quali en quali numeric
#fonction qui change pour une variable les noms des modalit� par un nombre :
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

#donn�e avec toute les variables num�rique :
datann[,c(2,3,4,5,7,8,9,11,16)]<-dataquali

#normalisation des variables 
scaleddata<-scale(datann[-17])
datann<-data.frame(scaleddata,datann[17])

#Echantillonage des donn�es
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


##Param�trage du r�seau de neurone
Nous allons param�trer cette m�thode gr�ce � de nombreux param�tres : 
 - hidden : sp�cifie le nombre de neuronnes dans la couche cach�
 - err.fct : fonction utilis� pour determiner le calcul de l'erreur
 - rep : Nombre d'entrainement du r�seau

 --------
La r�gularisation permet d'�viter le sur ou le sous ajustement des donn�es d'entrainement.Elle consiste � modifier la fonction d'erreur du r�seau en p�nalisant les poids importants par l'ajout d'un terme d'erreur suppl�mentaire. Cependant si la constante de r�gularisation est trop importante cela entraine un sous ajustement, et si elle est trop peu importante alors cela entraine un sur-ajustement. Car cela augmente la r�ussite de g�n�ralisation. 
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

#hiden=3 : le reseau a 1 couche cach� de 3 neurones

```

Cette fonction utilise d'autre param�tres comme : 
 - startweights : contient la valeurs de d�part des poids (par d�faut une initialisation al�atoire)
 
Repr�sentation graphique du r�seau de neurones s�l�ctionn� : 

```{r}
plot(nn)

```


##Pr�diction de la variable cible y

Voici la distribution de la variable y pr�dit par le r�seau de neurone :

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

Voici le taux d'erreur de cette pr�diction :
```{r}
#tx d'erreur :
tx<-(m[1,2])/length(ytest)
tx
```





