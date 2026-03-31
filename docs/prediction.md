# Étapes de la prediction

Ce document détaille la phase de prédiction implémentée dans `scripts/logreg_predict.py`, avec le rôle de chaque étape.

## 1. Entrée CLI et configuration

Fonction : `parse_command_line_arguments()`

Paramètres traités :
- `dataset_csv_path` : CSV de test.
- `trained_parameter_json_file_path` : fichier de poids appris (`weights.json`).
- `--out` / `-o` : chemin de sortie du CSV final.
- `--analysis-log` : active les logs détaillés.

Pourquoi cette étape existe :
- imposer explicitement le couple `dataset + paramètres` ;
- éviter l'usage d'un mauvais fichier de poids.

## 2. Chargement des paramètres entraînés

Fonction : `load_house_classifier_parameters(...)`

Sous-étapes :
1. Ouverture du JSON de paramètres.
2. Lecture de `thetas`, `mu`, `sigma`, `features`, `inv_house_map`.
3. Conversion de `thetas` en `numpy.ndarray`.
4. Conversion des clés de `inv_house_map` en `int`.

Sorties :
- matrice de poids ;
- statistiques de normalisation ;
- mapping `code -> maison` ;
- ordre exact des features attendu.

Pourquoi cette étape existe :
- garantir la compatibilité stricte avec le modèle entraîné ;
- préserver l'ordre des colonnes utilisé pendant le training.

## 3. Chargement et validation du dataset de test

Fonction : `load_observations(dataset_csv_path, discipline_names)`

Sous-étapes :
1. Lecture du CSV test.
2. Vérification de la présence de `Index`.
3. Vérification de la présence de toutes les features attendues.
4. Extraction de la liste `Index` (pour l'export).
5. Construction de `X_test` dans l'ordre `discipline_names`.

Pourquoi cette étape existe :
- prévenir les prédictions silencieusement incorrectes ;
- garantir un CSV de sortie aligné avec les index d'entrée.

## 4. Standardisation côté test

Fonction : `standardize_discipline_scores(...)`

Sous-étapes :
1. Conversion de `mu` et `sigma` en `numpy.float`.
2. Protection contre division par zéro :
   `sigma_safe = np.where(sigma == 0, 1.0, sigma)`.
3. Copie `float` du DataFrame test.
4. Imputation des valeurs manquantes avec les moyennes du training.
5. Conversion en `numpy`.
6. Transformation : `X_std = (X - mu) / sigma_safe`.

Pourquoi cette étape existe :
- placer les données test dans le même espace numérique que le training ;
- éviter l'échec sur features constantes ou valeurs manquantes.

## 5. Ajout du biais

Dans `main()` :
- construction de `X_test_bias = hstack([ones, X_std])`.

Pourquoi cette étape existe :
- respecter la même convention de représentation que durant l'entraînement.

## 6. Calcul des prédictions

Fonction : `predict_house_names(...)`

Sous-étapes :
1. `scores = X_test_bias.dot(thetas.T)` (un score par maison).
2. Clipping des scores dans `[-500, 500]`.
3. Conversion en probabilités avec sigmoid.
4. Sélection de la classe finale via `argmax(axis=1)`.
5. Mapping des codes vers les noms de maison.

Pourquoi cette étape existe :
- comparer les classifieurs one-vs-all sur chaque échantillon ;
- limiter les débordements numériques avant `exp`.

## 7. Génération de `houses.csv`

Dans `main()` :
1. Construction d'un DataFrame :
   - `Index`
   - `Hogwarts House`
2. Écriture via `to_csv(..., index=False)`.

Pourquoi cette étape existe :
- produire exactement le format attendu pour l'évaluation DSLR.

## 8. Logs d'analyse optionnels

Composant : `AnalysisPredictLogger`

Branches activées avec `--analysis-log` :
- affichage des données brutes ;
- affichage de la standardisation ;
- affichage de la matrice avec biais ;
- affichage des scores, probabilités, classes et labels finaux.

Pourquoi cette étape existe :
- rendre la phase de prédiction entièrement traçable pendant la soutenance.

## 9. Gestion d'erreur

Dans `main()` :
- bloc global `try/except` ;
- message CLI unique en cas d'erreur.

Pourquoi cette étape existe :
- fournir un diagnostic utilisateur clair sans trace technique verbeuse.
