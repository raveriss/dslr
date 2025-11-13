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
        -mu : moyenne utilisees pour normaliser
        -sigma : ecart-types utilises pour normaliser
        -inv_house_map :mapping label = maison

    Args:
        fichier(str) :chemin vers le fichier JSON

    Returns:
        thetas(numpy.ndarray) : matrice des poids
        mu (list[float] : moyenne colonne par colonne
        sigma (list[float]) :ecart-type colonne par colonne
        inv_house_map (dict) : nom des differente maison
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

    return thetas, mu, sigma, inv_house_map

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
    return numeric_cols

def load_and_prepare_data_test(path):
    """
    Lis le fichier dataset_test.csv

    Arg :
        chemin du fichier dataset_test.csv
    Return :
        X_test_df (panda.dataFrame): Contient que les colonne de matieres avec un eleve par ligne
    """
    df = pd.read_csv(path)
    # get_numeric_features(df) renvoie la liste des matières (colonnes de notes).
    features = get_numeric_features(df)
    # reset_index(drop=True) réindexe les élèves de 0 à m-1
    X_test_df = df[features].reset_index(drop=True)

    return X_test_df

def normalize_test_features(X_test_df, mu, sigma):

    return X_test_norm

def main():
    try:
        # Parser les arguments
        args = parse_args()
        # charger les paramettre apris
        thetas, mu, sigma, inv_house_map = load_weights(args.weights)
        # print(f"thetas = {thetas}, mu = {mu}, sigma = {sigma}, inv_house = {inv_house_map}")
        # charger le dataset de test
        X_test_df = load_and_prepare_data_test(args.input_csv)
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

        print(f"→ Fichier de prediction enregistres dans {args.out}")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()
