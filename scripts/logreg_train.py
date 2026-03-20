import argparse
import json

import numpy as np
import pandas as pd


def parse_command_line_arguments():
    """
    Parse les arguments de la ligne de commande pour logreg_train.py.

    Returns:
        Namespace: contient les attributs suivants
          - input_csv_path (str): chemin vers dataset_train.csv
          - learning_rate (float): taux d'apprentissage
          - iteration_count (int): nombre d'iterations de gradient descent
          - output_weights_path (str): chemin du fichier de sortie pour les poids
    """
    argument_parser = argparse.ArgumentParser(
        description="Entrainer un classifieur logistique one-vs-all et sauvegarder les poids."
    )
    argument_parser.add_argument(
        "input_csv_path",
        help="Chemin vers le fichier d'entrainement (dataset_train.csv)."
    )
    argument_parser.add_argument(
        "--alpha", "-a",
        dest="learning_rate",
        type=float,
        default=0.01,
        help="Taux d'apprentissage pour la descente de gradient (defaut : 0.01)."
    )
    argument_parser.add_argument(
        "--iterations", "-n",
        dest="iteration_count",
        type=int,
        default=1000,
        help="Nombre d'iterations de gradient descent (defaut : 1000)."
    )
    argument_parser.add_argument(
        "--out", "-o",
        dest="output_weights_path",
        default="weights.json",
        help="Fichier de sortie pour les poids entraines (defaut : 'weights.json')."
    )
    return argument_parser.parse_args()


def get_numeric_feature_columns(data_frame):
    """
    Identifie et retourne les colonnes numeriques correspondant aux matieres.

    Args:
        data_frame (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de colonnes numeriques, excluant 'Index'.
    """
    numeric_column_names = []
    for column_name in data_frame.columns:
        column_kind = data_frame[column_name].dtype.kind
        if column_kind in ("i", "f"):
            numeric_column_names.append(column_name)

    if "Index" in numeric_column_names:
        numeric_column_names.remove("Index")

    return numeric_column_names


def load_and_prepare_training_data(input_csv_path):
    """
    Lit le CSV d'entrainement et prepare X, y pour l'entrainement.

    Returns:
        training_feature_frame (pandas.DataFrame): features numeriques
        house_label_indices (list[int]): labels encodes (0-3)
        house_label_by_name (dict[str, int]): mapping maison -> label
        house_name_by_label (dict[int, str]): mapping label -> maison
        feature_names (list[str]): noms des matieres
    """
    training_data_frame = pd.read_csv(input_csv_path)

    training_data_frame = training_data_frame.dropna(
        subset=["Hogwarts House"] + get_numeric_feature_columns(training_data_frame)
    )

    feature_names = get_numeric_feature_columns(training_data_frame)
    training_feature_frame = training_data_frame[feature_names].reset_index(drop=True)

    house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house_label_by_name = {house_name: index for index, house_name in enumerate(house_names)}
    house_name_by_label = {label: house_name for house_name, label in house_label_by_name.items()}
    house_label_indices = training_data_frame["Hogwarts House"].map(house_label_by_name).tolist()

    return (
        training_feature_frame,
        house_label_indices,
        house_label_by_name,
        house_name_by_label,
        feature_names,
    )


def normalize_feature_matrix(training_feature_frame):
    """
    Centre et reduit les features.

    Returns:
        normalized_feature_matrix (numpy.ndarray): matrice normalisee
        feature_means (list[float]): moyennes de chaque colonne
        feature_standard_deviations (list[float]): ecart-types de chaque colonne
    """
    raw_feature_matrix = training_feature_frame.values.astype(float)
    feature_means = raw_feature_matrix.mean(axis=0)
    feature_standard_deviations = raw_feature_matrix.std(axis=0, ddof=0)
    normalized_feature_matrix = (
        raw_feature_matrix - feature_means
    ) / feature_standard_deviations

    return (
        normalized_feature_matrix,
        feature_means.tolist(),
        feature_standard_deviations.tolist(),
    )


def compute_sigmoid(values):
    """Fonction sigmoide."""
    return 1.0 / (1.0 + np.exp(-values))


def train_one_vs_all_classifier(
    training_matrix_with_bias,
    house_label_indices,
    learning_rate,
    iteration_count,
):
    """
    Entraine un classifieur logistique one-vs-all.

    Args:
        training_matrix_with_bias (numpy.ndarray): matrice normalisee avec biais
        house_label_indices (numpy.ndarray): labels entiers 0, 1, 2 ou 3
        learning_rate (float): taux d'apprentissage
        iteration_count (int): nombre d'iterations

    Returns:
        numpy.ndarray: matrice des poids de dimension (K, n+1)
    """
    student_count, feature_count_with_bias = training_matrix_with_bias.shape
    unique_house_labels = np.unique(house_label_indices)
    house_count = len(unique_house_labels)
    model_weights = np.zeros((house_count, feature_count_with_bias))

    for house_label in unique_house_labels:
        current_house_weights = np.zeros(feature_count_with_bias)
        is_current_house = (house_label_indices == house_label).astype(float)

        for _ in range(iteration_count):
            predicted_probabilities = compute_sigmoid(
                training_matrix_with_bias.dot(current_house_weights)
            )
            weight_gradient = (1 / student_count) * training_matrix_with_bias.T.dot(
                predicted_probabilities - is_current_house
            )
            current_house_weights -= learning_rate * weight_gradient

        model_weights[int(house_label), :] = current_house_weights

    return model_weights


def main():
    try:
        command_line_arguments = parse_command_line_arguments()
        (
            training_feature_frame,
            house_label_indices,
            house_label_by_name,
            house_name_by_label,
            feature_names,
        ) = load_and_prepare_training_data(command_line_arguments.input_csv_path)

        (
            normalized_feature_matrix,
            feature_means,
            feature_standard_deviations,
        ) = normalize_feature_matrix(training_feature_frame)

        student_count = normalized_feature_matrix.shape[0]
        training_matrix_with_bias = np.hstack(
            [np.ones((student_count, 1)), normalized_feature_matrix]
        )

        model_weights = train_one_vs_all_classifier(
            training_matrix_with_bias,
            house_label_indices,
            command_line_arguments.learning_rate,
            command_line_arguments.iteration_count,
        )

        output_payload = {
            "thetas": model_weights.tolist(),
            "mu": feature_means,
            "sigma": feature_standard_deviations,
            "features": feature_names,
            "house_map": house_label_by_name,
            "inv_house_map": house_name_by_label,
        }

        with open(command_line_arguments.output_weights_path, "w") as output_file:
            json.dump(output_payload, output_file)

        print(
            f"→ Poids et parametres enregistres dans {command_line_arguments.output_weights_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
