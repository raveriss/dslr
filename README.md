# DSLR (42) - DataScience x Logistic Regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-dans%20requirements-013243?logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-dans%20requirements-150458?logo=pandas&logoColor=white) ![matplotlib](https://img.shields.io/badge/matplotlib-dans%20requirements-orange?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI%2BPHBhdGggZD0iTTQgMTdWNyIvPjxwYXRoIGQ9Ik00IDE3aDE2Ii8%2BPHBhdGggZD0iTTYgMTVsMy00IDMgMiA0LTYgMiAzIi8%2BPC9nPjxnIGZpbGw9IiNmZmZmZmYiPjxjaXJjbGUgY3g9IjYiIGN5PSIxNSIgcj0iMS4yIi8%2BPGNpcmNsZSBjeD0iOSIgY3k9IjExIiByPSIxLjIiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEzIiByPSIxLjIiLz48Y2lyY2xlIGN4PSIxNiIgY3k9IjciIHI9IjEuMiIvPjxjaXJjbGUgY3g9IjE4IiBjeT0iMTAiIHI9IjEuMiIvPjwvZz48L3N2Zz4%3D) ![scikit--learn](https://img.shields.io/badge/scikit--learn-dans%20requirements%20%28benchmark%29-F7931E?logo=scikitlearn&logoColor=white)

</div>


Projet de classification multi-classe pour le sujet **DSLR** de 42.
Le but est de reconstruire un "Sorting Hat" avec une **régression logistique one-vs-all** implémentée sans fonctions "heavy-lifting" interdites par le sujet.

## Table des matières

- [1. Vue d'ensemble](#1-vue-densemble)
- [2. Objectifs du projet](#2-objectifs-du-projet)
- [3. Contexte pédagogique (42 / IA / ML)](#3-contexte-pédagogique-42--ia--ml)
- [4. Quick start (3 minutes)](#4-quick-start-3-minutes)
- [Documentation](#documentation)
- [5. Prérequis](#5-prérequis)
- [6. Installation](#6-installation)
- [7. Utilisation](#7-utilisation)
- [8. Scripts obligatoires du sujet DSLR](#8-scripts-obligatoires-du-sujet-dslr)
- [9. Entrées / sorties importantes](#9-entrées--sorties-importantes)
- [10. Commandes Make](#10-commandes-make)
- [11. Structure du projet](#11-structure-du-projet)
- [12. Tests, qualité et outils de dev](#12-tests-qualité-et-outils-de-dev)
- [13. Conformité au sujet DSLR (checklist)](#13-conformité-au-sujet-dslr-checklist)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Bonus / améliorations possibles](#15-bonus--améliorations-possibles)
- [16. Stack technique](#16-stack-technique)
- [17. Ressources](#17-ressources)
- [18. Licence](#18-licence)
- [19. Auteur](#19-auteur)

## 1. Vue d'ensemble

Ce dépôt contient :
- des scripts d'exploration de données (`describe`, `histogram`, `scatter_plot`, `pair_plot`) ;
- un entraînement de régression logistique multi-classe (`logreg_train`) ;
- une prédiction (`logreg_predict`) qui génère `houses.csv`.

Le flux principal est :
`comprendre les données -> visualiser -> entraîner -> prédire -> comparer`.

### 1.1 Pipeline détaillé

Le détail du pipeline est maintenant documenté dans `docs/` :
- [Pipeline global](docs/pipeline.md)
- [Étapes du training](docs/training.md)
- [Étapes de la prédiction](docs/prediction.md)

## 2. Objectifs du projet

- Implémenter une analyse descriptive sans `DataFrame.describe()`.
- Répondre aux 3 questions de visualisation imposées par le sujet.
- Implémenter une régression logistique **one-vs-all**.
- Utiliser la **descente de gradient** pour l'entraînement.
- Générer `houses.csv` au format attendu.

## 3. Contexte pédagogique (42 / IA / ML)

Ce projet fait partie du cursus 42 autour de l'IA/ML :
- lecture et nettoyage d'un dataset ;
- visualisation pour guider la sélection de features ;
- classification supervisée multi-classe.

Le sujet impose une partie technique, mais aussi une capacité à expliquer les notions (mean/std/quartiles, normalisation, one-vs-all, etc.) pendant la soutenance.

## 4. Quick start (3 minutes)

```bash
# 1) Cloner
 git clone git@github.com:Sycourbi/dslr.git
 cd dslr

# 2) Installer l'environnement
 make

# 3) Pipeline minimum
 make describe
 make train
 make predict
```

Résultats attendus :
- `make describe` affiche les stats en console.
- `make train` crée `weights.json`.
- `make predict` crée `houses.csv`.

## Documentation

- [Pipeline global](docs/pipeline.md)
- [Étapes du training](docs/training.md)
- [Étapes de la prédiction](docs/prediction.md)

## 5. Prérequis

- `python3` (version minimale officielle : `[À compléter]`)
- `make`
- accès shell Linux/macOS (ou équivalent)

Dépendances Python installées depuis `requirements.txt` :
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (utile pour le benchmark, pas pour l'algorithme "from scratch")

## 6. Installation

### Option A (recommandée)

```bash
make
```

Cette commande exécute `make install` :
- création de `.venv` ;
- installation des dépendances via `pip install -r requirements.txt`.


## 7. Utilisation

### 7.1 Exploration des données

```bash
make describe
make histogram
make scatter
make pair
```

Sorties générées :
- `visuals/histogram.png`
- `visuals/scatter.png`
- `visuals/pair_plot.png`

Format par défaut des visuels :
- `1920x1080` (`16:9`) pour `histogram`, `scatter` et `pair_plot` (affichage homogène sur écran PC portable).

### 7.1.1 Correspondance commandes -> graphiques

| Commande | Graphique généré | Aperçu |
|---|---|---|
| `make histogram` | `histogram` | ![Histogram example](docs/assets/histogram.png) |
| `make scatter` | `scatter plot` | ![Scatter plot example](docs/assets/scatter.png) |
| `make pair` | `pair plot` | ![Pair plot example](docs/assets/pair_plot.png) |
| `make animate` | `logreg_train_weights` | ![Train example](docs/assets/logreg_train_weights.gif) |
| `make kiviat` | `kiviat_house_discipline_weights` |![Kiviat example](docs/assets/kiviat_house_discipline_weights.png) |


### 7.2 Entraînement et prédiction

Le pipeline minimum (`make train` puis `make predict`) est déjà montré dans le **Quick start**.
Cette section regroupe surtout les variantes d'exécution utiles pour l'analyse.

### 7.2.1 Mode analyse détaillée (verbose)

```bash
make analysis_log_train
make analysis_log_predict
```

Sorties générées :
- `weights_training.json` (poids issus du dataset d'analyse)
- `houses_training.csv` (prédictions sur dataset d'analyse)
- logs détaillés en console (gradients, scores, probabilités, etc.)

### 7.3 Exemple en ligne de commande (sans Make)

```bash
.venv/bin/python scripts/logreg_train.py datasets/dataset_train.csv --alpha 0.01 --iterations 1000 --out weights.json
.venv/bin/python scripts/logreg_predict.py datasets/dataset_test.csv weights.json --out houses.csv
```

## 8. Scripts obligatoires du sujet DSLR

| Script | Statut sujet | Question/objectif | Entrée principale | Sortie principale |
|---|---|---|---|---|
| `scripts/describe.py` | `Obligatoire` | Afficher `count/mean/std/min/25%/50%/75%/max` des features numériques | `dataset_train.csv` | Affichage console |
| `scripts/histogram.py` | `Obligatoire` | Trouver un cours avec distribution homogène entre maisons | `dataset_train.csv` | `visuals/histogram.png` |
| `scripts/scatter_plot.py` | `Obligatoire` | Trouver deux features similaires | `dataset_train.csv` | `visuals/scatter.png` |
| `scripts/pair_plot.py` | `Obligatoire` | Visualiser les paires pour choisir les features du modèle | `dataset_train.csv` | `visuals/pair_plot.png` |
| `scripts/logreg_train.py` | `Obligatoire` | Entraîner la régression logistique multi-classe one-vs-all via gradient descent | `dataset_train.csv` | `weights.json` |
| `scripts/logreg_predict.py` | `Obligatoire` | Prédire et générer le fichier de rendu | `dataset_test.csv` + `weights.json` | `houses.csv` |

## 9. Entrées / sorties importantes

### Fichiers d'entrée

- `datasets/dataset_train.csv`
  - contient la cible `Hogwarts House` (pour l'entraînement)
- `datasets/dataset_test.csv`
  - utilisé pour la prédiction

### Fichiers de sortie

- `weights.json`
  - paramètres du modèle entraîné (`thetas`, `mu`, `sigma`, `features`, mapping des classes)
- `houses.csv`
  - format attendu :

```csv
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
...
398,Ravenclaw
399,Ravenclaw
```

- `visuals/histogram.png`, `visuals/scatter.png`, `visuals/pair_plot.png`
  - graphiques produits par les scripts d'exploration
- `visuals/logreg_train_weights.gif`
  - animation de l'évolution des poids pendant l'entraînement
- `visuals/kiviat_house_discipline_weights.png`
  - visualisation radar des poids par maison
- `weights_training.json`, `houses_training.csv`
  - artefacts du mode `analysis_log_*`

## 10. Commandes Make

| Commande | Rôle | Sortie / effet principal |
|---|---|---|
| `make` / `make all` | Alias d'installation | exécute `install` |
| `make install` | Crée/valide `.venv` + installe `requirements.txt` | environnement Python prêt |
| `make describe` | Statistiques descriptives | affichage console |
| `make histogram` | Histogrammes par matière/maison | `visuals/histogram.png` |
| `make scatter` | Nuages de points | `visuals/scatter.png` |
| `make pair` | Pair plot global | `visuals/pair_plot.png` |
| `make train` | Entraînement logreg one-vs-all | `weights.json` |
| `make predict` | Prédiction avec poids entraînés | `houses.csv` |
| `make analysis_log_train` | Entraînement verbose sur dataset d'analyse | `weights_training.json` + logs détaillés |
| `make analysis_log_predict` | Prédiction verbose sur dataset d'analyse | `houses_training.csv` + logs détaillés (nécessite `weights_training.json`) |
| `make animate` | Génère une animation de l'évolution des poids | `visuals/logreg_train_weights.gif` |
| `make kiviat` | Génère un radar des poids par maison | `visuals/kiviat_house_discipline_weights.png` |
| `make clean` | Supprime artefacts générés | supprime `weights.json`, `houses.csv`, `visuals/`, caches Python (conserve `weights_training.json` et `houses_training.csv`) |
| `make fclean` | Nettoyage complet | `clean` + suppression `.venv` |
| `make re` | Réinitialisation environnement | `fclean` puis `all` |
| `make help` | Aide intégrée | affichage des targets |

## 11. Structure du projet

```text
dslr/
├── datasets/
│   ├── dataset_train.csv
│   └── dataset_test.csv
├── docs/
│   ├── assets/
│   ├── pipeline.md
│   ├── training.md
│   └── prediction.md
├── scripts/
│   ├── describe.py
│   ├── histogram.py
│   ├── scatter_plot.py
│   ├── pair_plot.py
│   ├── logreg_train.py
│   └── logreg_predict.py
├── scikit/
│   └── benchmark_sklearn_vs_mine.py
├── Makefile
├── requirements.txt
├── dslr.subject.pdf
└── README.md
```

## 12. Tests, qualité et outils de dev

### Ce qui est présent

- Script de comparaison optionnel :

```bash
.venv/bin/python scikit/benchmark_sklearn_vs_mine.py \
  datasets/dataset_train.csv \
  datasets/dataset_test.csv \
  houses.csv
```

Ce script entraîne un modèle scikit-learn, génère `houses_sklearn.csv`, puis compare avec `houses.csv`.


## 13. Conformité au sujet DSLR (checklist)

### Obligatoire sujet

- `describe` présent : `Oui`
- `histogram` présent : `Oui`
- `scatter_plot` présent : `Oui`
- `pair_plot` présent : `Oui`
- `logreg_train` présent : `Oui`
- `logreg_predict` présent : `Oui`
- Logique multi-classe `one-vs-all` : `Oui` (dans `logreg_train.py`)
- Descente de gradient : `Oui` (dans `logreg_train.py`)
- Génération de `houses.csv` : `Oui` (dans `logreg_predict.py`)

### Points d'évaluation importants à connaître

- Pas de fonctions interdites qui font tout le travail dans `describe`.
- Format de sortie `houses.csv` strict.
- Objectif de précision à la soutenance : minimum `98%` (selon le sujet/grille).
- Bonus évalués seulement si le mandatory est parfait.

## 14. Troubleshooting

- Erreur `No such file or directory: 'weights.json'` lors de `make predict`
  - Cause : `make train` non exécuté (ou `weights.json` supprimé).
  - Fix : relancer `make train`, puis `make predict`.

- `python3-venv is missing`
  - Le `Makefile` tente automatiquement un fallback avec `virtualenv` utilisateur.

- Aucune image générée
  - Vérifier que le dossier de sortie existe (`visuals/`) ou passer `-o <dossier>`.

## 15. Bonus / améliorations possibles

`Bonus sujet` (liste du PDF) :
- ajouter d'autres métriques dans `describe`
- implémenter une descente stochastique du gradient
- implémenter d'autres algorithmes d'optimisation (GD par lots/GD par mini-lots/etc.) nombre d'échantillons

## 16. Stack technique

- Langage : `Python`
- Data : `pandas`, `numpy`
- Visualisation : `matplotlib`
- Référence de comparaison : `scikit-learn` (script optionnel)
- Orchestration locale : `Makefile`

## 17. Ressources

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

## 18. Licence

MIT License.

## 19. Auteurs


- **Sylvanna Courbis** — [LinkedIn](https://www.linkedin.com/in/sylvanna-courbis-7626b63a7/) · [GitHub](https://github.com/Sycourbi)
- **Rafael Verissimo** — [LinkedIn](https://www.linkedin.com/in/verissimo-rafael/) · [GitHub](https://github.com/raveriss)
