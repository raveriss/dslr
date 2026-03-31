# Étapes du training

Ce document détaille la phase d'entraînement implémentée dans `scripts/logreg_train.py`, avec le rôle de chaque étape.

## 1. Entrée CLI et configuration

Fonction : `parse_command_line_arguments()`

Paramètres traités :
- `input_csv_path` : CSV d'entraînement.
- `--alpha` / `-a` : taux d'apprentissage.
- `--iterations` / `-n` : nombre d'itérations.
- `--out` / `-o` : chemin de sortie JSON.
- `--analysis-log` : active les logs détaillés.

Pourquoi cette étape existe :
- stabiliser le contrat d'exécution du script ;
- permettre de régler l'optimisation sans modifier le code.

## 2. Chargement et préparation des données

Fonction : `load_and_prepare_dataset(input_csv_path)`

Sous-étapes :
1. Lecture du dataset via `pd.read_csv`.
2. Détection des features numériques via `get_discipline_names`.
3. Retrait de la colonne technique `Index`.
4. Suppression des lignes incomplètes sur la cible et les features :
   `dropna(subset=["Hogwarts House"] + discipline_names)`.
5. Construction de `X_train` avec un ordre de colonnes stable.
6. Définition d'un ordre canonique des maisons :
   `["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]`.
7. Encodage texte -> entier de la cible (`y_train`).

Pourquoi cette étape existe :
- éviter les incohérences de schéma entre entraînement et prédiction ;
- garantir un mapping de classes stable dans le temps.

## 3. Standardisation

Fonction : `standardize_disciplines_scores(students_disciplines_scores)`

Sous-étapes :
1. Conversion DataFrame -> `numpy.float`.
2. Calcul des moyennes `mu` par feature.
3. Calcul des écarts-types `sigma` par feature (`ddof=0`).
4. Transformation : `X_std = (X - mu) / sigma`.
5. Retour de `X_std`, `mu`, `sigma` (utiles aussi pour la prédiction).

Pourquoi cette étape existe :
- mettre toutes les features à une échelle comparable ;
- améliorer la stabilité de la descente de gradient.

## 4. Ajout du biais

Dans `main()` :
- construction de `X_bias = hstack([ones, X_std])`.

Pourquoi cette étape existe :
- introduire l'intercept dans le modèle linéaire logistique.

## 5. Entraînement one-vs-all

Fonction : `fit_one_vs_rest_house_classifier(...)`

Sous-étapes globales :
1. Initialiser une matrice de poids `house_disciplines_weights`.
2. Boucler sur chaque maison (classification binaire par maison).
3. Pour chaque maison :
   - initialiser `current_house_weights` à 0 ;
   - construire `y_binary = (y_train == current_house_code)`.
4. Boucler sur `iteration_count` :
   - `p = sigmoid(X_bias.dot(current_house_weights))`
   - `error = p - y_binary`
   - `grad_sum = X_bias.T.dot(error)`
   - `gradient = grad_sum / students_count`
   - `current_house_weights -= alpha * gradient`
5. Stocker les poids de la maison entraînée dans la matrice globale.

Pourquoi cette étape existe :
- traiter un problème multi-classe avec un classifieur logistique binaire ;
- optimiser les poids via gradient descent batch.

## 6. Logs d'analyse optionnels

Composant : `AnalysisLogger`

Branches activées uniquement avec `--analysis-log` :
- affichage des scores initiaux ;
- traces par maison ;
- traces par itération (probabilités, erreurs, gradients, poids).

Pourquoi cette étape existe :
- faciliter l'explication en soutenance ;
- auditer le comportement numérique du training.

## 7. Sérialisation du modèle

Dans `main()` :
- création d'un `trained_parameter_bundle` contenant :
  - `thetas`
  - `mu`
  - `sigma`
  - `features`
  - `house_map`
  - `inv_house_map`
- sauvegarde JSON via `json.dump`.

Pourquoi cette étape existe :
- conserver tout le contexte nécessaire à la phase de prédiction ;
- éviter tout recalcul des statistiques de normalisation au moment de prédire.

## 8. Gestion d'erreur

Dans `main()` :
- bloc global `try/except` ;
- message CLI unique en cas d'échec.

Pourquoi cette étape existe :
- produire des erreurs lisibles côté utilisateur ;
- éviter les crashes non contrôlés.
