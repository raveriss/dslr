import argparse
import json

import numpy as np
import pandas as pd


def parse_command_line_arguments():
    """
    Parse les arguments de la ligne de commande pour logreg_predict.py.

    Returns:
        Namespace: contient les attributs suivants
          - input_csv_path (str): chemin vers dataset_test.csv
          - weights_file_path (str): fichier json contenant les poids
          - output_csv_path (str): chemin du fichier de sortie a genere
    """
    argument_parser = argparse.ArgumentParser(
        description="Predit la maison de chaque eleve a partir d'un dataset test et du fichier json"
    )
    argument_parser.add_argument(
        "input_csv_path",
        help="Chemin vers le fichier dataset_test.csv (sans colonne House)"
    )
    argument_parser.add_argument(
        "weights_file_path",
        help="Fichier JSON contenant les poids appris (creer par logreg_train.py)"
    )
    argument_parser.add_argument(
        "--out", "-o",
        dest="output_csv_path",
        default="houses.csv",
        help="Fichier de sortie contenant les predictions"
    )
    return argument_parser.parse_args()


def load_model_parameters(weights_file_path):
    """
    Charge le fichier JSON contenant :
        - thetas : matrice de poids (K x n+1)
        - mu : moyenne des matieres utilisees pour normaliser
        - sigma : ecart-types utilises pour normaliser
        - features : nom de toutes les matieres
        - inv_house_map : mapping label -> maison

    Args:
        weights_file_path (str): chemin vers le fichier JSON

    Returns:
        model_weights (numpy.ndarray): matrice des poids
        feature_means (list[float]): moyenne colonne par colonne
        feature_standard_deviations (list[float]): ecart-type colonne par colonne
        house_name_by_label (dict[int, str]): nom des differentes maisons
        feature_names (list[str]): noms des matieres
    """
    with open(weights_file_path, "r") as weights_file:
        model_payload = json.load(weights_file)

    model_weights = np.array(model_payload["thetas"])
    feature_means = model_payload["mu"]
    feature_standard_deviations = model_payload["sigma"]
    house_name_by_label = {
        int(label): house_name
        for label, house_name in model_payload["inv_house_map"].items()
    }
    feature_names = model_payload["features"]

    return (
        model_weights,
        feature_means,
        feature_standard_deviations,
        house_name_by_label,
        feature_names,
    )


def get_numeric_feature_columns(data_frame):
    """
    Identifie et retourne les colonnes numeriques correspondant aux matieres.

    Args:
        data_frame (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de colonnes numeriques, excluant 'Index' et 'Hogwarts House'.
    """
    numeric_column_names = []
    for column_name in data_frame.columns:
        column_kind = data_frame[column_name].dtype.kind
        if column_kind in ("i", "f"):
            numeric_column_names.append(column_name)

    if "Index" in numeric_column_names:
        numeric_column_names.remove("Index")
    if "Hogwarts House" in numeric_column_names:
        numeric_column_names.remove("Hogwarts House")

    return numeric_column_names


def load_test_dataset(input_csv_path, feature_names):
    """
    Lit le fichier dataset_test.csv.

    Args:
        input_csv_path (str): chemin du fichier dataset_test.csv
        feature_names (list[str]): colonnes attendues depuis weights.json

    Returns:
        student_index_list (list[int]): liste des index du dataset test
        test_feature_frame (pandas.DataFrame): colonnes de matieres dans le bon ordre
    """
    test_data_frame = pd.read_csv(input_csv_path)

    if "Index" not in test_data_frame.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier test.")

    missing_feature_names = [
        feature_name for feature_name in feature_names
        if feature_name not in test_data_frame.columns
    ]
    if missing_feature_names:
        raise ValueError(f"Colonnes manquantes dans le fichier test: {missing_feature_names}")

    student_index_list = test_data_frame["Index"].tolist()
    test_feature_frame = test_data_frame[feature_names].copy()

    return student_index_list, test_feature_frame


def normalize_test_features(test_feature_frame, feature_means, feature_standard_deviations):
    """
    Normalise les features de test avec les parametres du train.

    Args:
        test_feature_frame (pandas.DataFrame): features brutes du dataset test
        feature_means (list[float]): moyenne des matieres calculee au train
        feature_standard_deviations (list[float]): ecart-type calcule au train

    Returns:
        numpy.ndarray: matrice normalisee
    """
    feature_means = np.array(feature_means, dtype=float)
    feature_standard_deviations = np.array(feature_standard_deviations, dtype=float)
    safe_standard_deviations = np.where(feature_standard_deviations == 0, 1.0, feature_standard_deviations)

    filled_test_feature_frame = test_feature_frame.astype(float).copy()
    for feature_index, column_name in enumerate(filled_test_feature_frame.columns):
        filled_test_feature_frame[column_name] = filled_test_feature_frame[column_name].fillna(
            feature_means[feature_index]
        )

    test_feature_matrix = filled_test_feature_frame.values.astype(float)
    normalized_test_feature_matrix = (test_feature_matrix - feature_means) / safe_standard_deviations
    return normalized_test_feature_matrix


def predict_houses(test_feature_matrix_with_bias, model_weights, house_name_by_label):
    """
    Predit la maison de chaque eleve a partir des features test et des poids appris.

    Args:
        test_feature_matrix_with_bias (numpy.ndarray): matrice test avec colonne biais
        model_weights (numpy.ndarray): matrice des poids
        house_name_by_label (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    class_scores = test_feature_matrix_with_bias.dot(model_weights.T)
    class_scores = np.clip(class_scores, -500, 500)
    class_probabilities = 1 / (1 + np.exp(-class_scores))
    predicted_label_indices = np.argmax(class_probabilities, axis=1)
    predicted_house_labels = [
        house_name_by_label[int(label_index)] for label_index in predicted_label_indices
    ]
    return predicted_house_labels


def main():
    try:
        command_line_arguments = parse_command_line_arguments()
        (
            model_weights,
            feature_means,
            feature_standard_deviations,
            house_name_by_label,
            feature_names,
        ) = load_model_parameters(command_line_arguments.weights_file_path)

        student_index_list, test_feature_frame = load_test_dataset(
            command_line_arguments.input_csv_path,
            feature_names
        )

        normalized_test_feature_matrix = normalize_test_features(
            test_feature_frame,
            feature_means,
            feature_standard_deviations
        )

        student_count = normalized_test_feature_matrix.shape[0]
        test_feature_matrix_with_bias = np.hstack(
            [np.ones((student_count, 1)), normalized_test_feature_matrix]
        )

        predicted_house_labels = predict_houses(
            test_feature_matrix_with_bias,
            model_weights,
            house_name_by_label
        )

        prediction_output_frame = pd.DataFrame(
            {"Index": student_index_list, "Hogwarts House": predicted_house_labels}
        )
        prediction_output_frame.to_csv(command_line_arguments.output_csv_path, index=False)
        print(
            f"→ Fichier de prediction enregistre dans {command_line_arguments.output_csv_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
