#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt


# Quel cours de Poudlard a une distribution de notes homogène entre les quatre maisons ?
def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Namespace: les arguments parsés (input_csv).
    """
    # argparse.ArgumentParser:
    # Je crée un nouvel analyseur d’arguments (ArgumentParser)
    parser = argparse.ArgumentParser(
        description="Afficher l'entête du tableau descriptif du dataset."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier CSV d'entrée (dataset_train.csv)."
    )
    # Nombre de bins pour l'histogramme
    parser.add_argument(
        "--bins", "-b",
        type=int,
        default=50,
        help="Nombre de bins pour l'histogramme (défaut : 50)."
    )
    # Dossier de sortie pour les images
    parser.add_argument(
        "--outdir", "-o",
        default="visuals",
        help="Dossier de sortie pour les PNG (défaut : 'visualization')."
    )
    return parser.parse_args()

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

def find_most_homogeneous(df, features, houses):
    best_feature = None
    best_score = float("inf")

    for feat in features:
        means = []
        for house in houses:
            vals = df[df["Hogwarts House"] == house][feat].dropna()
            if len(vals) > 0:
                means.append(vals.mean())

        if len(means) > 1:
            score = max(means) - min(means)
            if score < best_score:
                best_score = score
                best_feature = feat

    return best_feature

def one_histogram(df, feature, houses, bins, outdir):
    """
    Trace tous les histogrammes des matières dans une grille 3×4 et sauvegarde
    le résultat dans un seul PNG.

    Args:
        df (pandas.DataFrame): Le dataset complet contenant 'Hogwarts House'.
        features (list of str): Liste des noms de colonnes numériques (matières).
        houses (array-like): Liste des quatre maisons.
        bins (int): Nombre de bins pour chaque histogramme.
        outdir (str): Dossier où enregistrer le fichier unique.
    """

    # 1. Créer une figure avec 3 lignes et 4 colonnes de sous-plots
    fig, ax = plt.subplots(figsize=(10, 6))

    for house in houses:
        vals = df[df["Hogwarts House"] == house][feature].dropna()
        ax.hist(vals, bins=bins, alpha=0.5, label=house)
        # b) Titres et labels
        # ax.set_title(feat)
        ax.set_xlabel(feature)
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Histogramme de {feature} par maison")
        ax.legend(fontsize='small')

    # 3. Ajuster la mise en page et ajouter un titre global
    plt.tight_layout()
    # 4. Enreg_scatteristrer le fichier unique
    outfile = os.path.join(outdir, "histogram.png")
    fig.savefig(outfile)
    plt.close(fig)
    print(f"→ histogram.png créé dans {outdir}/")


def main():
    try:
        # Parser les arguments
        args = parse_args()
        # Charger le dataset
        df = pd.read_csv(args.input_csv)

        # Récupérer la liste des maisons
        houses = df["Hogwarts House"].dropna().unique()
        # Identifier les colonnes numériques (features)
        features = get_numeric_features(df)
        best_feature = find_most_homogeneous(df, features, houses)
        # Préparer le dossier de sortie 'visualization'
        os.makedirs(args.outdir, exist_ok=True)
        # Trace et sauvegarde un histogramme superposé pour chaque feature.
        # all_histograms(df, features, houses, args.bins, args.outdir)
        one_histogram(df, best_feature, houses, args.bins, args.outdir)
    
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    
if __name__ == "__main__":
    main()