import argparse
import pandas as pd
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sklearn on dataset_train.csv, predict dataset_test.csv, and compare with my houses.csv"
    )
    parser.add_argument("train_csv", help="Path to dataset_train.csv")
    parser.add_argument("test_csv", help="Path to dataset_test.csv")
    parser.add_argument("mine_csv", help="Path to your houses.csv")
    parser.add_argument("--out", default="houses_sklearn.csv", help="Output predictions file")
    return parser.parse_args()


def get_numeric_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Index" in numeric_cols:
        numeric_cols.remove("Index")
    return numeric_cols


def save_predictions(indexes, labels, out_path):
    out_df = pd.DataFrame({
        "Index": indexes,
        "Hogwarts House": labels
    })
    out_df.to_csv(out_path, index=False)


def load_predictions(path):
    df = pd.read_csv(path)
    return dict(zip(df["Index"], df["Hogwarts House"]))


def compare_predictions(mine_path, skl_path):
    mine = load_predictions(mine_path)
    skl = load_predictions(skl_path)

    common_indexes = sorted(set(mine.keys()) & set(skl.keys()))
    same = 0
    diffs = []

    for idx in common_indexes:
        if mine[idx] == skl[idx]:
            same += 1
        else:
            diffs.append((idx, mine[idx], skl[idx]))

    total = len(common_indexes)
    agreement = same / total if total else 0.0

    print("\n=== Comparison ===")
    print(f"Common indexes : {total}")
    print(f"Same predictions: {same}")
    print(f"Agreement      : {agreement:.4f}")

    if diffs:
        print("\nFirst disagreements:")
        for idx, mine_label, skl_label in diffs[:10]:
            print(f"- Index {idx}: mine={mine_label} | sklearn={skl_label}")
    else:
        print("\nNo disagreement found.")


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if "Hogwarts House" not in train_df.columns:
        raise ValueError("dataset_train.csv must contain 'Hogwarts House'")
    if "Index" not in test_df.columns:
        raise ValueError("dataset_test.csv must contain 'Index'")

    features = get_numeric_features(train_df)

    # Keep only rows with known target in train
    train_df = train_df.dropna(subset=["Hogwarts House"]).copy()

    # Fill NaN with train means
    train_means = train_df[features].mean()
    train_df[features] = train_df[features].fillna(train_means)
    test_df[features] = test_df[features].fillna(train_means)

    # Normalization from train only
    mu = train_df[features].mean()
    sigma = train_df[features].std(ddof=0).replace(0, 1)

    X_train = ((train_df[features] - mu) / sigma).values
    y_train = train_df["Hogwarts House"].values

    X_test = ((test_df[features] - mu) / sigma).values
    test_indexes = test_df["Index"].values

    # One-vs-all / One-vs-Rest
    clf = OneVsRestClassifier(LogisticRegression(max_iter=5000))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    save_predictions(test_indexes, y_pred, args.out)
    print(f"Sklearn predictions saved to: {args.out}")

    compare_predictions(args.mine_csv, args.out)


if __name__ == "__main__":
    main()