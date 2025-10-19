# 🎬 LLMovie

## 🧠 Objectif du projet

Ce projet a pour but de **développer un système de recommandation de films fonctionnant entièrement en local**, sans appel à une API externe ni stockage cloud.  
Il repose sur des **techniques de filtrage collaboratif**, d’analyse de notes utilisateurs et de **réduction de dimension** (via *Truncated SVD*).  

Le but est de pouvoir recommander des films :
-  à partir des préférences d’un utilisateur (basées sur ses notations),
-  à partir d’un film donné (trouver les films les plus similaires),
-  ou donner un film aléatoire.

---

## 📦 Installation des données

### 1️⃣ Télécharger le dataset MovieLens

Ce projet s’appuie sur le jeu de données [**MovieLens**](https://grouplens.org/datasets/movielens/latest/).  

Ce dataset contient :
- 33,000,000 notes d'utilisateurs 
- 86,000 films 
- 330,975 utilisateurs.

Les derniers ajouts dans ce dataset date du 09/2018. Les films les plus récents datent de 2024. 

Exemple :  
Télécharge le fichier `ml-latest-small.zip` (ou `ml-latest.zip` pour la version complète).

### 2️⃣ Préparer la structure du projet

Crée un dossier `dataset/` à la racine du projet :

```css
📁 /
┣ 📁 dataset/
┃ ┣ ratings.csv
┃ ┣ movies.csv
┃ ┗ ...
┣ main.py
┣ README.md
┗ requirements.txt
```


Place les fichiers du dataset MovieLens (ex. `ratings.csv`, `movies.csv`) dans ce dossier.

---

### 3️⃣ (Optionnel) Ajouter ton profil Letterboxd

Pour une recommandation plus spécifique tu peux ajouter **tes données Letterboxd** :

1. Va sur ton profil Letterboxd (ex : `https://letterboxd.com/ton_nom/`).
2. Clique sur `Settings` → `Import & Export`.
3. Clique sur **"Export Your Data"** pour télécharger ton fichier `.zip`.
4. Met le `.zip` dans le dossier `dataset/`.

Ainsi Le projet peut ainsi utiliser ton propre historique de films pour produire des recommandations personnalisées.

---

## ⚙️ Fonctionnement du projet

Ce projet s’exécute **entièrement en local** :  
Aucune requête externe n’est envoyée.  
Les calculs et les recommandations sont faits directement sur ton ordinateur, à partir des fichiers CSV présents dans `dataset/`.

### 🧩 Pipeline générale

TODO

---

## 🧮 Méthodes de recommandation

Le système utilise trois approches complémentaires :

### 1️⃣ Recommandation basée sur un utilisateur

Tout d'abord on cherche a calculer le bias utilisateurs et le biais films

Film:
$$
b_i = \frac{\sum_{u \in U_i} (r_{ui} - \mu)}{|U_i| + \lambda} 
$$

Utilisateur:
$$
b_u = \frac{\sum_{i \in I_u} (r_{ui} - \mu - b_i)}{|I_u| + \lambda} 
$$



$$
r_{ui} \approx \mu + b_u + b_i
$$


---

### 2️⃣ Recommandation basée sur un film (item-item)

TODO

---

### 3️⃣​ Recommandation aléatoire

Pour cette recommandation on demande seulement au LLM de nous recommander un film.

---

## 🚀 Exécution

TODO
