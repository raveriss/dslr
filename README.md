# DSLR Project 42 AI

Ce projet a pour objectif de recréer le "chapeau magique" (Sorting Hat) de Poudlard à l'aide d'un algorithme de régression logistique multi-classe implémenté "from scratch".

## Structure du projet

```
dslr/
├─ datasets/
│   ├─ dataset_train.csv
│   └─ dataset_test.csv
├─ scripts/
│   ├─ describe.py
│   ├─ histogram.py
│   ├─ scatter_plot.py
│   ├─ pair_plot.py
│   ├─ logreg_train.py
│   └─ logreg_predict.py
├─ requirement.txt
├─ Makefile
└─ README.md
```

## Prérequis

* Python 3.8+

## Installation

1. Cloner le dépôt & naviguer dans le dossier :

   ```bash
   git clone <url_du_repo>
   cd dslr
   ```
2. Installer les dépendances :

   ```bash
   make
   ```

## Usage des scripts

Chaque script se trouve dans le dossier `scripts/` et comporte une aide intégrée :

```bash
.venv/bin/python scripts/<nom_du_script>.py --help
```

### 1. `describe.py`

**Objectif** : Calculer et afficher les statistiques descriptives de chaque colonne numérique d'un fichier CSV, sans utiliser `DataFrame.describe()`.


**Fonctionnalités** :

* Identification des colonnes numériques
* Calcul manuel de :

  * `count` combien d’élèves ont une note enregistrée(exclut les vides)
  * `mean` la moyenne des notes
  * `std` l’écart-type (mesure de la dispersion)
  * `Min / Max` la note la plus basse et la plus haute
  * Quartiles `25%`, `50%` (médiane), `75%`
* Affichage formaté en tableau dans la console

**Exemple d'utilisation** :

```bash
.venv/bin/python scripts/describe.py datasets/dataset_train.csv
```

* Le script lit `dataset_train.csv` (jeu d'entraînement), calcule les statistiques pour chaque colonne numérique et affiche un tableau comme :

| Statistique | Astronomy | History |
| ----------- | --------- | ------- |
| Count       | 395       | 395     |
| Mean        | 66.5      | 72.1    |
| Std         | 10.2      | 12.4    |
| Min         | 30        | 35      |
| 25%         | 58        | 62      |
| 50%         | 67        | 71      |
| 75%         | 75        | 84      |
| Max         | 100       | 100     |


### 2. `histogram.py`

**Objectif** : Tracer et analyser les distributions des notes pour chaque matière, afin d'identifier la matière avec la distribution la plus homogène, et sauvegarder les histogrammes générés.

```bash
.venv/bin/python scripts/histogram.py datasets/dataset_train.csv
```
### 3. `scatter_plot.py`

**Objectif** : Générer un nuage de points entre deux features pour visualiser les relations et identifier des corrélations potentielles.

```bash
.venv/bin/python scripts/scatter_plot.py datasets/dataset_train.csv
```

### 4. `pair_plot.py`

**Objectif** :  Afficher la matrice complète de scatter plots pour toutes les paires de colonnes numériques, facilitant la sélection des variables pour la modélisation.

```bash
.venv/bin/python scripts/pair_plot.py datasets/dataset_train.csv
```
### 5. `logreg_train.py`

**Objectif** : Utiliser `dataset_train.csv` pour entraîner les modèles de régression logistique multi-classe (one-vs-all) via descente de gradient et sauvegarder les poids.

```bash
.venv/bin/python scripts/logreg_train.py datasets/dataset_train.csv
```
### 6. `logreg_predict.py`

**Objectif** : Utiliser `dataset_test.csv` et les poids entraînés pour prédire la maison de chaque élève en calculant les probabilités, puis générer le fichier de soumission `houses.csv`.

```bash
.venv/bin/python scripts/logreg_predict.py datasets/dataset_test.csv weights.json
```

### Différence entre `dataset_train.csv` et `dataset_test.csv`

* **`dataset_train.csv`** : jeu de données d'entraînement complet

  * Contient les **features** (notes, attributs élèves, etc.) **et** la colonne cible `Hogwarts House`
  * Utilisé pour :

    1. **Exploration** (describe, visualisations)
    2. **Entraînement** des modèles de régression logistique (logreg\_train.py)

* **`dataset_test.csv`** : jeu de données de test sans la maison assignée

  * Contient uniquement les mêmes **features** que le train, **sans** la colonne `Hogwarts House`
  * Utilisé pour :

    1. **Prédiction** (logreg\_predict.py) en chargeant les poids obtenus
    2. Génération du fichier de soumission `houses.csv`

### Comparer avec scikit 
```bash
.venv/bin/python scikit/benchmark_sklearn_vs_mine.py datasets/dataset_train.csv datasets/dataset_test.csv houses.csv
```
**Objectif** : Utiliser scikit pour entrainer, predire puis creer houses_sklearn.csv et le comparer avec houses.csv

Interprétation

Common indexes : 400
→ les deux fichiers contiennent bien les 400 mêmes élèves

Same predictions: 400
→ pour chaque élève, la maison prédite est la même

Agreement : 1.0000
→ 100 % d’accord

No disagreement found.
→ aucune différence entre houses.csv et houses_sklearn.csv

## Lien utils

https://scikit-learn.org/stable/

## Licence

MIT
