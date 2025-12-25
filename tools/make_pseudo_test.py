import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Créer un pseudo dataset de test à partir du dataset d'entraînement "
                    "en supprimant la colonne 'Hogwarts House'."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le dataset d'entraînement (dataset_train.csv)."
    )
    parser.add_argument(
        "--out", "-o",
        default="datasets/pseudo_test.csv",
        help="Chemin du fichier de sortie (défaut : datasets/pseudo_test.csv)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Vérifier que le fichier d'entrée existe
    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Fichier introuvable : {args.input_csv}")

    # Charger le dataset
    df = pd.read_csv(args.input_csv)

    # Vérifier que la colonne cible existe
    if "Hogwarts House" not in df.columns:
        raise ValueError("La colonne 'Hogwarts House' n'existe pas dans le fichier fourni.")

    # Supprimer la colonne des maisons
    df_pseudo_test = df.drop(columns=["Hogwarts House"])

    # Sauvegarder le nouveau fichier
    df_pseudo_test.to_csv(args.out, index=False)

    print("✔ Pseudo dataset de test créé avec succès")
    print(f"→ Entrée : {args.input_csv}")
    print(f"→ Sortie : {args.out}")
    print(f"→ Colonnes conservées : {list(df_pseudo_test.columns)}")


if __name__ == "__main__":
    main()
