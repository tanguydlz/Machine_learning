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

Nous allons voir le protocole exp�rimentale, puis nous allons le d�crire plus en d�tail le d�roulement de l'analyse r�alis� ainsi que les conclusions.

##Protocole exp�rimental



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
```

De plus, pour la v�rification et la validation du meilleur reseau de neurone, nous d�coupons nos donn�es en trois �chantillons. Nous r�alisons un premier d�coupage dans les donn�es pour obtenir un �chantillon d'apprentissage qui est constitu� de 80% de la base et un �chantillon test qui est constitu� de 20% de la base. Puis nous re-d�coupons l'�chantillon d'apprentissage en deux, pour r�cup�rer un �chantillon de validation qui est constitu� de 30% de l'�chantillon d'apprentissage. 

```{r}
#Echantillonage des donn�es
ech = sort(sample(nrow(datann), nrow(datann)*.8))

train = datann[ech, ]
test = datann[-ech, ]

#Definition variables explicatives/variable cible de test
Xtest = test[, -17]
ytest = test[, 17]

#D�coupage de l'�chantillon d'apprentissage en ech de validation : 
#Echantillonage des donn�es
ech2 = sort(sample(nrow(train), nrow(train)*.7))

valid = train[-ech2, ]
train = train[ech2, ]

#Definition variables explicatives/variable cible 
Xtrain = train[, -17]
ytrain = train[, 17]
Xvalid = valid[, -17]
yvalid = valid[, 17]

```


##Param�trage du r�seau de neurone

Nous allons r�aliser un r�seau de neurones � l'aide de la fonction "neutralnet" qui permet de r�aliser tout type de r�seau de neurones.
Nous allons param�trer cette m�thode gr�ce � de nombreux param�tres : 
 - hidden : sp�cifie le nombre de neuronnes dans la couche cach�
 - err.fct : fonction utilis� pour determiner le calcul de l'erreur (ce : entropie crois�e, sse : somme des erreurs au carr�)
 - linear.output = FALSE : Ne pas r�aliser de regression lin�aire
 - algorithm : contient une chaine d�finissant le type d'algorithme, par d�faut vaut "rprop+" qui signifie r�tropropagation r�siliente avec retour de poids
 - act.fct : permet de lisser le r�sultat du produit crois� des neurones et des poids, par d�faut "logistic" qui se rapporte � la fonction logistique.

Cette fonction utilise d'autre param�tres comme : 
 - startweights : contient la valeurs de d�part des poids (par d�faut une initialisation al�atoire)
 - rep : Nombre d'entrainement du r�seau (pas utiliser pour �viter le sur-apprentissage)

Dans le cas d'une variable y cat�gorielle binaire, nous choisissons comme fonction d'erreur l'entropie croiss�e.

Premier mod�le : 

```{r}
nn=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE)
attributes(nn)
nn$result.matrix

```

Validation du mod�le : 

```{r}
prob <- predict(nn,Xvalid)
pred <- ifelse(prob>0.5, 1, 0)
pred

```

Voici la matrice de confusion entre les valeurs pr�dite et les valeurs test et le taux d'erreur :  
```{r}
m<-table(pred,yvalid)
m
#tx d'erreur :
tx<-(m[1,2])/length(yvalid)
tx
```

Le taux d'erreur de ce premier mod�le est de 8,5%.

Deuxi�me mod�le : 
```{r}
nn2=neuralnet(ytrain~., data = Xtrain, hidden=10, err.fct="ce",linear.output = FALSE,likelihood = TRUE)

prob2 <- predict(nn2,yvalid)
pred2 <- ifelse(prob2>0.5, 1, 0)

m2<-table(pred2,yvalid)
tx2<-(m2[1,2])/length(yvalid)
tx2

```
Le mod�le deux est plus performant avec le "likelihood" (taux d'erreur = 8,1% < 8,5%).

Troixi�me mod�le : augmentation du nombre de couche de la couche cach� de 10 � 17 (nombre de variable d'entr�e)
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

Quatri�me mod�le : augmentation du nombre de couche de la couche cach� de 17 � 34 (deux fois le nombre de variable d'entr�e)
```{r}
nn4=neuralnet(ytrain~., data = Xtrain, hidden=34, err.fct="ce",linear.output = FALSE)

prob4 <- predict(nn4,valid)
pred4 <- ifelse(prob4>0.5, 1, 0)

m4<-table(pred4,yvalid)
tx4<-(m4[1,2])/length(yvalid)
tx4

```

Taux d'erreur un peu �lev�, donc moins int�ressant (7,6%<7,9%)

On garde la pr�diction n�3 pour le nombre de neurones de la couche cach�. Et nous d�cidons de garder 17 neurones dans la couche cach�.

Cinqui�me mod�le : On utilise une couche cach� compos� de 34 neurones et on passe l'argument "likelihood" � vrai.
```{r}
nn5<-neuralnet(ytrain~., data = Xtrain, hidden=34, err.fct="ce",linear.output = FALSE, likelihood = TRUE)

prob5 <- predict(nn5,valid)
pred5 <- ifelse(prob5>0.5, 1, 0)

m5<-table(pred5,yvalid)
tx5<-(m5[1,2])/length(yvalid)
tx5

```

Cette fois ci le taux d'erreur est plus �lev� (8,0%)

Nous testons un dernier mod�le.

Sixi�me mod�le : Nous augumentons encore une fois le nombre de neurones de la couche cach�e (3 fois le nombre de variable d'entr�e)
```{r}
nn6<-neuralnet(ytrain~., data = Xtrain, hidden=51, err.fct="ce",linear.output = FALSE, likelihood = TRUE)

prob6 <- predict(nn6,valid)
pred6 <- ifelse(prob6>0.5, 1, 0)

m6<-table(pred6,yvalid)
tx6<-(m6[1,2])/length(yvalid)
tx6

```
Cette fois le taux d'erreur est de 7,8% ce qui est faiblement plus �lev� que pour le quatri�me mod�le (7,6%). Mais ce mod�le utilise un plus grand nombre de neurone ce qui peut engendrer du sur-apprentissage. 

Avec la library "neutralnn" et la variable cible y correspondant � une classification binaire, il n'est pas possible de param�trer le r�seau de neurone � l'infinie. Les param�trages pour une telle variable sont faible. Le param�tre "algorithme" ne fonctionne qu'avec "rprop +", la fonction d'erreur ne peut pas �tre modifier et la demande d'une r�gression logistique est impossible. Nous aurions pu changer les poids les variables d'entr�e, mais le choix de l'al�atoire nous pariassait plus juste.
N�anmoins, avec les r�seaux de neurones que nous avons construit nous avons un taux d'erreur plut�t faible, ce qui nous paraissait �tre un bon r�sultat.Pour confirmer cette avis, nous d�cidons de comparer les courbes ROC des six mod�les.

courbe ROC : 
```{r}
all_pred = data.frame(pred, pred2, pred3, pred4, pred5, pred6)
colnames(all_pred) = c("nn","nn2","nn3","nn4","nn5","nn6")
colAUC(all_pred, yvalid, plotROC = TRUE)
abline(0,1, col = "blue")

```

A l'aide de la Courbe ROC ainsi que du taux d'erreur, nous consid�rons le meilleur r�seau de neurone �tant le quatri�me mod�le construit. C'est a dire que nous utilisons 17 neuronnes cach� et que nous pla�ons � vrai l'argument "likelihood". C'est avec ce mod�le que nous allons v�rifier la bonne pr�diction des valeurs test.

##Pr�diction de la variable cible y

Repr�sentation graphique du r�seau de neurones s�l�ctionn� : 

```{r}
plot(nn4)

```

Voici la distribution de la variable y pr�dite par le r�seau de neurone :

```{r}
prob <- predict(nn4,test)
pred <- ifelse(prob>0.5, 1, 0)
pred

```

Voici la matrice de confusion entre les valeurs pr�dite et les valeurs test : 
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
Le taux d'erreur est de 7,1%.

courbe ROC repr�sentant : 
```{r}
colnames(pred)="Reseau de neurone"
colAUC(pred, ytest, plotROC = TRUE)
abline(0,1, col = "blue")

```

L'air sous la courbe repr�sentant le reseau de neurones est sup�rieur � celle de la courbe repr�sentant l'al�toire.

