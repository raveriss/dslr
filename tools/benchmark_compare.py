import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark: entraîne ton modèle + sklearn, prédit sur pseudo_test, compare résultats."
    )
    p.add_argument("--train", default="datasets/dataset_train.csv", help="Chemin vers dataset_train.csv")
    p.add_argument("--pseudo", default="datasets/pseudo_test.csv", help="Chemin vers pseudo_test.csv (sans maisons)")
    p.add_argument("--weights", default="weights.json", help="Fichier weights.json (sortie de logreg_train)")
    p.add_argument("--mine-out", default="houses_mine.csv", help="Sortie de tes prédictions")
    p.add_argument("--skl-out", default="houses_sklearn.csv", help="Sortie des prédictions sklearn")

    p.add_argument("--alpha", type=float, default=0.01, help="Learning rate pour ton modèle")
    p.add_argument("--iterations", type=int, default=1000, help="Nb itérations gradient descent pour ton modèle")
    p.add_argument("--seed", type=int, default=42, help="Seed pour split/ordre reproducible (sklearn)")

    p.add_argument(
        "--make-pseudo",
        action="store_true",
        help="Si activé: régénère pseudo_test.csv depuis train en supprimant Hogwarts House."
    )
    return p.parse_args()


def run_cmd(cmd_list):
    # Affiche la commande pour debug soutenance-friendly
    print("→", " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)


def make_pseudo_from_train(train_path, pseudo_path):
    df = pd.read_csv(train_path)
    if "Hogwarts House" not in df.columns:
        raise ValueError("dataset_train.csv doit contenir la colonne 'Hogwarts House'")
    df = df.drop(columns=["Hogwarts House"])
    os.makedirs(os.path.dirname(pseudo_path), exist_ok=True)
    df.to_csv(pseudo_path, index=False)
    print(f"✔ pseudo_test généré: {pseudo_path}")


def safe_sigma(sigma):
    # évite division par zéro
    out = []
    for s in sigma:
        out.append(s if s != 0 else 1.0)
    return np.array(out, dtype=float)


def load_truth_from_train(train_path):
    df = pd.read_csv(train_path)
    if "Index" not in df.columns or "Hogwarts House" not in df.columns:
        raise ValueError("dataset_train.csv doit contenir 'Index' et 'Hogwarts House'")
    truth = {}
    for _, row in df.iterrows():
        truth[int(row["Index"])] = row["Hogwarts House"]
    return truth


def load_predictions_csv(path):
    df = pd.read_csv(path)
    # on accepte soit "Index", soit première colonne
    if "Index" not in df.columns:
        df.columns = ["Index", "Hogwarts House"]
    # on accepte soit "Hogwarts House", soit seconde colonne
    if "Hogwarts House" not in df.columns:
        df.columns = ["Index", "Hogwarts House"]
    preds = {}
    for _, row in df.iterrows():
        preds[int(row["Index"])] = str(row["Hogwarts House"])
    return preds


def manual_accuracy(preds, truth):
    # preds et truth: dict Index -> label
    total = 0
    ok = 0
    for idx, t in truth.items():
        if idx in preds:
            total += 1
            if preds[idx] == t:
                ok += 1
    return (ok / total) if total else 0.0, ok, total


def manual_agreement(pred_a, pred_b):
    total = 0
    same = 0
    for idx, a in pred_a.items():
        if idx in pred_b:
            total += 1
            if a == pred_b[idx]:
                same += 1
    return (same / total) if total else 0.0, same, total


def sklearn_train_and_predict(train_path, pseudo_path, weights_path, out_path, seed):
    # On récupère mu/sigma + mapping depuis tes weights (ainsi sklearn utilise EXACTEMENT la même normalisation et mapping)
    with open(weights_path, "r") as f:
        w = json.load(f)

    mu = np.array(w["mu"], dtype=float)
    sigma = safe_sigma(np.array(w["sigma"], dtype=float))

    # mapping label->maison (clé JSON = string)
    inv_house_map = w.get("inv_house_map")
    if inv_house_map is None:
        raise ValueError("weights.json doit contenir 'inv_house_map'")

    inv = {}
    for k, v in inv_house_map.items():
        inv[int(k)] = v

    # Charger train complet, dropna comme toi (on copie ta logique: on enlève lignes NaN sur target+features)
    df = pd.read_csv(train_path)
    if "Hogwarts House" not in df.columns:
        raise ValueError("dataset_train.csv doit contenir Hogwarts House")

    # features = colonnes numériques hors Index
    numeric_cols = []
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind in ("i", "f"):
            numeric_cols.append(col)
    if "Index" in numeric_cols:
        numeric_cols.remove("Index")

    df = df.dropna(subset=["Hogwarts House"] + numeric_cols).reset_index(drop=True)

    # y : on reconstruit un mapping stable basé sur inv_house_map (maison->label)
    # inv_house_map: label->maison
    house_to_label = {}
    for k, house in inv.items():
        house_to_label[house] = k

    y = []
    for house in df["Hogwarts House"].tolist():
        y.append(int(house_to_label[house]))
    y = np.array(y, dtype=int)

    X = df[numeric_cols].values.astype(float)
    X_norm = (X - mu) / sigma

    # Entraîner sklearn OVR
    clf = LogisticRegression(max_iter=5000, random_state=seed)
    clf.fit(X_norm, y)

    # Charger pseudo_test (sans maison)
    dfp = pd.read_csv(pseudo_path)
    if "Index" not in dfp.columns:
        raise ValueError("pseudo_test.csv doit contenir la colonne 'Index'")

    dfp = dfp.dropna(subset=numeric_cols).reset_index(drop=True)
    
    Xp = dfp[numeric_cols].values.astype(float)
    Xp_norm = (Xp - mu) / sigma

    y_pred = clf.predict(Xp_norm)

    # Écrire CSV au format: Index,Hogwarts House
    rows = []
    for i in range(len(y_pred)):
        idx = int(dfp.loc[i, "Index"])
        rows.append((idx, inv[int(y_pred[i])]))

    out_df = pd.DataFrame(rows, columns=["Index", "Hogwarts House"])
    out_df.to_csv(out_path, index=False)
    print(f"✔ prédictions sklearn écrites: {out_path}")


def main():
    args = parse_args()

    if args.make_pseudo:
        make_pseudo_from_train(args.train, args.pseudo)

    # 1) Entraînement (TES scripts)
    run_cmd([
        sys.executable, "scripts/logreg_train.py",
        args.train,
        "--alpha", str(args.alpha),
        "--iterations", str(args.iterations),
        "--out", args.weights
    ])

    # 2) Prédiction (TES scripts)
    run_cmd([
        sys.executable, "scripts/logreg_predict.py",
        args.pseudo,
        args.weights,
        "--out", args.mine_out
    ])

    # 3) sklearn benchmark
    sklearn_train_and_predict(
        train_path=args.train,
        pseudo_path=args.pseudo,
        weights_path=args.weights,
        out_path=args.skl_out,
        seed=args.seed
    )

    # 4) Comparaisons
    truth = load_truth_from_train(args.train)
    mine = load_predictions_csv(args.mine_out)
    skl = load_predictions_csv(args.skl_out)

    mine_acc, mine_ok, mine_tot = manual_accuracy(mine, truth)
    skl_acc, skl_ok, skl_tot = manual_accuracy(skl, truth)
    agree, same, tot = manual_agreement(mine, skl)

    print("\n=== Résultats ===")
    print(f"Mine accuracy vs vérité : {mine_acc:.4f} ({mine_ok}/{mine_tot})")
    print(f"SKL  accuracy vs vérité : {skl_acc:.4f} ({skl_ok}/{skl_tot})")
    print(f"Accord Mine vs SKL      : {agree:.4f} ({same}/{tot})")

    # Quelques exemples de désaccord (pour debug)
    shown = 0
    print("\nExemples de désaccord (Index: vérité | mine | skl) :")
    for idx in sorted(truth.keys()):
        if idx in mine and idx in skl and mine[idx] != skl[idx]:
            print(f"- {idx}: {truth[idx]} | {mine[idx]} | {skl[idx]}")
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("- Aucun (parfaitement identiques sur les Index communs)")

    print("\n✔ Terminé.")


if __name__ == "__main__":
    main()
