# Modélisation de Performances Sportives (FPL Predictor)

## 1. Contexte du Projet

Ce projet est né d'une volonté d'appliquer l'analyse de données et l'algèbre linéaire à un problème concret de prise de décision : l'optimisation d'une équipe sur la Fantasy Premier League (FPL). 
Plutôt que de me fier uniquement à l'instinct pour évaluer la forme d'un joueur, gérer les semaines de "blanks" (matchs reportés) ou analyser les surperformances par rapport à l'average global, j'ai décidé de modéliser ces paramètres mathématiquement. L'objectif est de prédire les points futurs d'un joueur grâce à une approche purement statistique.

## 2. Approche Mathématique

Le cœur de l'algorithme repose sur une **régression linéaire multiple**. Le modèle cherche à trouver le vecteur de coefficients $\beta$ qui pondère au mieux chaque variable statistique.

L'équation matricielle du modèle s'écrit :
$Y = X\beta + \epsilon$

Pour trouver les meilleurs coefficients, le script implémente le calcul matriciel exact de l'estimateur des moindres carrés ordinaires (MCO) :
$\hat{\beta} = (X^T X)^{-1} X^T y$

## 3. Architecture des Données (Matrice X)

Les variables explicatives (colonnes de la matrice $X$) intègrent des statistiques avancées :
* **Points de la journée précédente** (Dynamique)
* **FDR (Fixture Difficulty Rating)** (Impact de l'adversité)
* **Écart à l'average** (Surperformance par rapport à la moyenne globale)
* **xG (Expected Goals) et xA (Expected Assists)** (Probabilités de concrétisation)

## 4. Implémentation Technique

Développé en Python, le projet s'articule autour de deux bibliothèques majeures :
* **`pandas` :** Utilisé pour le Data Cleaning, notamment pour la gestion stricte des "blanks" (valeurs manquantes) afin de garantir une matrice saine avant les calculs.
* **`numpy` :** Utilisé pour l'algèbre linéaire (transposition, multiplication et inversion de matrices) afin d'extraire les coefficients $\beta$ à partir de zéro, sans passer par une boîte noire algorithmique.

## 5. Exécution

```bash
# Création et activation de l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate

# Installation des dépendances
pip install pandas numpy

# Lancement de la prédiction
python fpl_prediction.py