import argparse
import os
import json

import pandas as pd
import numpy as np

def parse_args():
    """
    Parse les arguments de la ligne de commande pour logreg_predict.py.

    Returns:
        Namespace: contient les attributs suivants
          - input_csv (str)      : chemin vers dataset_test.csv
          - weights              : fichier json contenant les poids
          - out (str)            : chemin du fichier de sortie a genere
    """
    parser = argparse.ArgumentParser(
        description="Predit la maison de chaque elleve a partir d'un dataset test et du fichier json"
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier dataset_test.csv (sans colonne House)"
    )
    parser.add_argument(
        "weights",
        help="Fichier JSON contenant les poids appris (creer par logreg_train.py)"
    )
    parser.add_argument(
        "--out", "-o",
        default="houses.csv",
        help="Fichier de sortie contenant les prediction"
    )
    return parser.parse_args()

def load_weights(fichier):
    """
    Charge le fichier JSON contenant : 
        -thetas : matrice de poids (K x n+1)
        -mu : moyenne des matieres utilisees pour normaliser
        -sigma : ecart-types utilises pour normaliser
        -features : nom de toute les matieres
        -inv_house_map :mapping label = maison

    Args:
        fichier(str) :chemin vers le fichier JSON

    Returns:
        thetas(numpy.ndarray) : matrice des poids
        mu (list[float] : moyenne colonne par colonne
        sigma (list[float]) :ecart-type colonne par colonne
        inv_house_map (dict[int, str]) : nom des differente maison
    """
    #ouvre et lire le JSON
    with open(fichier, 'r') as f:
        data = json.load(f)
    #extrait les diferrent elements
    thetas = np.array(data["thetas"])
    mu = data["mu"]
    sigma = data["sigma"]
    #reconvetie les cles en int
    inv_house_map = {int(k): v for k, v in data["inv_house_map"].items()}
    features = data["features"]

    return thetas, mu, sigma, inv_house_map, features

def get_numeric_features(df):
    """
    Identifie et retourne les colonnes numériques correspondant aux matières.

    Args:
        df (pandas.DataFrame): Données complètes du fichier.

    Returns:
        list of str: Liste des noms de colonnes numériques, excluant 'Index'.
    """
    # Initialisation de la liste des colonnes numériques
    numeric_cols = []
    # Parcours de chaque colonne du DataFrame
    for col in df.columns:
        # df[col] renvoie une Series contenant toutes les valeurs de la colonne
        # df[col].dtype.kind renvoie un code à un caractère pour le type :
        #   'i' pour int, 'f' pour float, 'O' pour object (texte), 'M' pour datetime, etc.
        kind = df[col].dtype.kind
        # On sélectionne uniquement les colonnes dont le type est entier ou flottant
        if kind in ("i", "f"):
            numeric_cols.append(col)
    # Exclusion de la colonne 'Index' si elle apparaît dans les numériques
    if "Index" in numeric_cols:
        numeric_cols.remove("Index")
    # Exclusion de la colonne 'House' si elle apparaît dans les numériques
    if "Hogwarts House" in numeric_cols:
        numeric_cols.remove("Hogwarts House")
    return numeric_cols

def load_and_prepare_data_test(path, features):
    """
    Lis le fichier dataset_test.csv

    Arg :
        path : chemin du fichier dataset_test.csv
        features : liste des colonnes attendues depuis weights.json
    Return :
        index_list (list[int]) : liste des index du dataset test
        X_test_df (pandas.DataFrame) : colonnes de matieres dans le bon ordre
    """
    df = pd.read_csv(path)

    if "Index" not in df.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier test.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier test: {missing}")

    index_list = df["Index"].tolist()
    X_test_df = df[features].copy()

    return index_list, X_test_df

def normalize_test_features(X_test_df, mu, sigma):
    """
    normalise les feature de data_testen utilisant les paramettre
        calculer pendant l'entraiment(mu et sigma)

    Arg :
        X_test_df(panda.dataFrame): Feature brute du data_test
        mu (list[float]): moyenne des matieres colonne calculer par train
        sigma (list[float]): ecart-type colonne calculer par train
    Return:
        X_test_norm(numpy.ndarray): Matrice normaliser
    """
    # Converti mu et sigma en array pour pouvoir faire les operation
    mu = np.array(mu, dtype=float)
    sigma = np.array(sigma, dtype=float)
    # sigma safe (évite division par 0)
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    
    # Convertir en float (DataFrame) et IMPUTER NaN colonne par colonne
    X_df = X_test_df.astype(float).copy()
    for j, col in enumerate(X_df.columns):
        # remplace NaN par la moyenne d'entraînement
        X_df[col] = X_df[col].fillna(mu[j])
    X = X_df.values.astype(float)
    # Normaliser chaque valeur: (x_ij - mu_j) / sigma_j
    # On soustrait la moyenne de la colonne (on centre),
    # On divise par l’écart-type (on réduit).
    X_norm = (X - mu) / sigma_safe
    return X_norm

def prediction_house(X_test_bias, thetas, inv_house_map):
    """
    Predit la maison de chaque eleve a partir de la matrice de features
    deja normalisee et les poids appris

    Arg:
        X_test_bias(numpy.ndarray)
        thetas(numpy.ndarray) : matrice des poids
        inv_house_map (dict[int, str]) : nom des differente maison

    Return: 
        predicted_house(list[str]):Liste des maison predi pour chaque eleve
    """
    #Calcule le score pour chaque eleve et chaque maison :
    # score[i, k] = kmatrice m x K (m = eleve et K = matiere)
    scores = X_test_bias.dot(thetas.T)
    #sigmoide pour convertir les scores en probabilite pour donner des score entre 0 et 1
    scores = np.clip(scores, -500, 500)
    probs = 1 / (1 + np.exp(-scores))
    # Selection de la maison avec la probabililte maximal
    pred_label = np.argmax(probs, axis=1)
    # Convertion de label numerique  en nom de maison
    prediction_house = [inv_house_map[int(k)] for k in pred_label]
    return prediction_house

def main():
    try:
        # Parser les arguments
        args = parse_args()
        # charger les paramettre apris
        thetas, mu, sigma, inv_house_map, features = load_weights(args.weights)
        # print(f"thetas = {thetas}, mu = {mu}, sigma = {sigma}, inv_house = {inv_house_map}")
        # charger le dataset de test
        index_list, X_test_df = load_and_prepare_data_test(args.input_csv, features)
        # print(f"test_df = {X_test_df}")
        # apliquer la meme normaliser que pour dataset_train
        X_test_norm = normalize_test_features(X_test_df, mu, sigma)
        # Ajouter la colonne biais
        # m = nombre d’élèves (lignes dans X_norm)
        m = X_test_norm.shape[0]
        # Ajouter la colonne biais
        # np.ones((m, 1)) = Crée une colonne de 1 (un 1 pour chaque élève)
        X_test_bias = np.hstack([np.ones((m, 1)), X_test_norm])
        # Predire la maison de chaque eleve
        predict_house = prediction_house(X_test_bias, thetas, inv_house_map)
        # Sauvegerder les prediction dans houses.csv
        df_out = pd.DataFrame({
            "Index": index_list,
            "Hogwarts House": predict_house
        })
        df_out.to_csv(args.out, index=False)
        print(f"→ Fichier de prediction enregistres dans {args.out}")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()


