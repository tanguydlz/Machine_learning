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

Nous allons voir le protocole expérimentale, puis nous allons le décrire plus en détail le déroulement de l'analyse réalisé ainsi que les conclusions.

##Protocole expérimental



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

Nous allons réaliser un réseau de neurones à l'aide de la fonction "neutralnet" qui permet de réaliser tout type de réseau de neurones.
Nous allons paramétrer cette méthode grâce à de nombreux paramètres : 
 - hidden : spécifie le nombre de neuronnes dans la couche caché
 - err.fct : fonction utilisé pour determiner le calcul de l'erreur (ce : entropie croisée, sse : somme des erreurs au carré)
 - linear.output = FALSE : Ne pas réaliser de regression linéaire
 - algorithm : contient une chaine définissant le type d'algorithme, par défaut vaut "rprop+" qui signifie rétropropagation résiliente avec retour de poids
 - act.fct : permet de lisser le résultat du produit croisé des neurones et des poids, par défaut "logistic" qui se rapporte à la fonction logistique.

Cette fonction utilise d'autre paramètres comme : 
 - startweights : contient la valeurs de départ des poids (par défaut une initialisation aléatoire)
 - rep : Nombre d'entrainement du réseau (pas utiliser pour éviter le sur-apprentissage)

Dans le cas d'une variable y catégorielle binaire, nous choisissons comme fonction d'erreur l'entropie croissée.

Premier modéle : 

```{r}
nn=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE)
attributes(nn)
nn$result.matrix

```

Validation du modèle : 

```{r}
prob <- predict(nn,Xvalid)
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

Le taux d'erreur de ce premier modéle est de 8,5%.

Deuxième modèle : 
```{r}
nn2=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE,likelihood = TRUE)

prob2 <- predict(nn2,yvalid)
pred2 <- ifelse(prob2>0.5, 1, 0)

m2<-table(pred2,yvalid)
tx2<-(m2[1,2])/length(yvalid)
tx2

```
Le modéle deux est plus performant avec le "likelihood" (taux d'erreur = 8,1% < 8,5%).

Troixième modèle : augmentation du nombre de couche de la couche caché de 10 à 17 (nombre de variable d'entrée)
```{r}
nn3=neuralnet(ytrain~., data = Xtrain, hidden=17, err.fct="ce",linear.output = FALSE)

prob3 <- predict(nn3,valid)
pred3 <- ifelse(prob3>0.5, 1, 0)
pred31<-pred3

m3<-table(pred3,yvalid)
tx3<-(m3[1,2])/length(yvalid)
tx3

```

Taux d'erreur un peu meilleur (7,9%<8,1%)

Quatrième modèle : augmentation du nombre de couche de la couche caché de 17 à 34 (deux fois le nombre de variable d'entrée)
```{r}
nn4=neuralnet(ytrain~., data = Xtrain, hidden=34, err.fct="ce",linear.output = FALSE)

prob4 <- predict(nn4,valid)
pred4 <- ifelse(prob4>0.5, 1, 0)

m4<-table(pred4,yvalid)
tx4<-(m4[1,2])/length(yvalid)
tx4

```

Taux d'erreur un peu élevé, donc moins intéressant (7,6%<7,9%)

On garde la prédiction n°3 pour le nombre de neurones de la couche caché. Et nous décidons de garder 17 neurones dans la couche caché.

Cinquième modèle : On utilise une couche caché composé de 34 neurones et on passe l'argument "likelihood" à vrai.
```{r}
nn5<-neuralnet(ytrain~., data = Xtrain, hidden=34, err.fct="ce",linear.output = FALSE, likelihood = TRUE)

prob5 <- predict(nn5,valid)
pred5 <- ifelse(prob5>0.5, 1, 0)

m5<-table(pred5,yvalid)
tx5<-(m5[1,2])/length(yvalid)
tx5

```

Cette fois ci le taux d'erreur est plus élevé (8,0%)

Nous testons un dernier modéle.

Sixième modèle : Nous augumentons encore une fois le nombre de neurones de la couche cachée (3 fois le nombre de variable d'entrée)
```{r}
nn6<-neuralnet(ytrain~., data = Xtrain, hidden=51, err.fct="ce",linear.output = FALSE, likelihood = TRUE)

prob6 <- predict(nn6,valid)
pred6 <- ifelse(prob6>0.5, 1, 0)

m6<-table(pred6,yvalid)
tx6<-(m6[1,2])/length(yvalid)
tx6

```
Cette fois le taux d'erreur est de 7,8% ce qui est faiblement plus élevé que pour le quatrième modèle (7,6%). Mais ce modèle utilise un plus grand nombre de neurone ce qui peut engendrer du sur-apprentissage. 

Avec la library "neutralnn" et la variable cible y correspondant à une classification binaire, il n'est pas possible de paramétrer le réseau de neurone à l'infinie. Les paramétrages pour une telle variable sont faible. Le paramètre "algorithme" ne fonctionne qu'avec "rprop +", la fonction d'erreur ne peut pas être modifier et la demande d'une régression logistique est impossible. Nous aurions pu changer les poids les variables d'entrée, mais le choix de l'aléatoire nous pariassait plus juste.
Néanmoins, avec les réseaux de neurones que nous avons construit nous avons un taux d'erreur plutôt faible, ce qui nous paraissait être un bon résultat.Pour confirmer cette avis, nous décidons de comparer les courbes ROC des six modèles.

courbe ROC : 
```{r}
all_pred = data.frame(pred, pred2, pred3, pred4, pred5, pred6)
colnames(all_pred) = c("nn","nn2","nn3","nn4","nn5","nn6")
colAUC(all_pred, yvalid, plotROC = TRUE)
abline(0,1, col = "blue")

```

A l'aide de la Courbe ROC ainsi que du taux d'erreur, nous considérons le meilleur réseau de neurone étant le quatrième modèle construit. C'est a dire que nous utilisons 17 neuronnes caché et que nous plaçons à vrai l'argument "likelihood". C'est avec ce modèle que nous allons vérifier la bonne prédiction des valeurs test.

##Prédiction de la variable cible y

Représentation graphique du réseau de neurones séléctionné : 

```{r}
plot(nn4)

```

Voici la distribution de la variable y prédite par le réseau de neurone :

```{r}
prob <- predict(nn4,test)
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
Le taux d'erreur est de 7,1%.

courbe ROC représentant : 
```{r}
colnames(pred)="Reseau de neurone"
colAUC(pred, ytest, plotROC = TRUE)
abline(0,1, col = "blue")

```

L'air sous la courbe représentant le reseau de neurones est supérieur à celle de la courbe représentant l'alétoire.

