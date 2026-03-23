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
          - output_parameter_path (str): chemin du fichier de sortie des parametres appris
    """
    argument_parser = argparse.ArgumentParser(
        description="Entrainer un classifieur logistique one-vs-all et sauvegarder ses parametres."
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
        dest="output_parameter_path",
        default="weights.json",
        help="Fichier de sortie des parametres appris (defaut : 'weights.json')."
    )
    return argument_parser.parse_args()


def get_numeric_subject_score_columns(dataset_table):
    """
    Identifie et retourne les colonnes numeriques correspondant aux matieres.

    Args:
        dataset_table (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de colonnes numeriques, excluant 'Index'.
    """
    numeric_column_names = []
    for column_name in dataset_table.columns:
        column_kind = dataset_table[column_name].dtype.kind
        if column_kind in ("i", "f"):
            numeric_column_names.append(column_name)

    if "Index" in numeric_column_names:
        numeric_column_names.remove("Index")

    return numeric_column_names


def load_and_prepare_training_dataset(input_csv_path):
    """
    Lit le CSV d'entrainement et prepare X, y pour l'entrainement.

    Returns:
        training_subject_score_table (pandas.DataFrame): notes numeriques par matiere
        target_house_code_array (numpy.ndarray): codes maisons attendus (0-3)
        house_code_by_name (dict[str, int]): mapping maison -> code maison
        house_name_by_code (dict[int, str]): mapping code maison -> maison
        subject_score_column_names (list[str]): noms des matieres
    """
    training_dataset_table = pd.read_csv(input_csv_path)
    numeric_subject_score_columns = get_numeric_subject_score_columns(training_dataset_table)

    training_dataset_table = training_dataset_table.dropna(
        subset=["Hogwarts House"] + numeric_subject_score_columns
    )

    subject_score_column_names = get_numeric_subject_score_columns(training_dataset_table)
    training_subject_score_table = (
        training_dataset_table[subject_score_column_names].reset_index(drop=True)
    )

    house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house_code_by_name = {house_name: code for code, house_name in enumerate(house_names)}
    house_name_by_code = {code: house_name for house_name, code in house_code_by_name.items()}
    target_house_code_array = training_dataset_table["Hogwarts House"].map(
        house_code_by_name
    ).to_numpy(dtype=int)

    return (
        training_subject_score_table,
        target_house_code_array,
        house_code_by_name,
        house_name_by_code,
        subject_score_column_names,
    )


def standardize_discipline_scores(training_discipline_score_table):
    """
    Centre et reduit les notes par matiere.
    Formule appliquee: (note_eleve_matiere - moyenne_matiere) / ecart_type_matiere.

    Returns:
        standardized_discipline_scores (numpy.ndarray): notes normalisees
        average_score_by_discipline (list[float]): moyenne de chaque matiere
        standard_deviation_by_discipline (list[float]): ecart-type de chaque matiere
    """
    discipline_scores_by_student = training_discipline_score_table.to_numpy(dtype=float)
    average_score_by_discipline = discipline_scores_by_student.mean(axis=0)
    standard_deviation_by_discipline = discipline_scores_by_student.std(axis=0, ddof=0)
    standardized_discipline_scores = (
        discipline_scores_by_student - average_score_by_discipline
    ) / standard_deviation_by_discipline

    return (
        standardized_discipline_scores,
        average_score_by_discipline.tolist(),
        standard_deviation_by_discipline.tolist(),
    )


def compute_sigmoid(linear_score_array):
    """Fonction sigmoide."""
    return 1.0 / (1.0 + np.exp(-linear_score_array))


def fit_one_vs_rest_house_classifier(
    standardized_scores_with_intercept,
    target_house_code_array,
    learning_rate,
    iteration_count,
):
    """
    Entraine un classifieur logistique one-vs-all.

    Args:
        standardized_scores_with_intercept (numpy.ndarray): notes normalisees + terme d'interception
        target_house_code_array (numpy.ndarray): codes maisons 0, 1, 2 ou 3
        learning_rate (float): taux d'apprentissage
        iteration_count (int): nombre d'iterations

    Returns:
        numpy.ndarray: tableau des coefficients de dimension (K, n+1)
    """
    student_count, predictor_count_with_intercept = standardized_scores_with_intercept.shape
    distinct_house_codes = np.unique(target_house_code_array)
    house_count = len(distinct_house_codes)
    house_coefficient_table = np.zeros((house_count, predictor_count_with_intercept))

    for current_house_code in distinct_house_codes:
        current_house_coefficients = np.zeros(predictor_count_with_intercept)
        is_student_in_current_house = (
            target_house_code_array == current_house_code
        ).astype(float)

        for _ in range(iteration_count):
            predicted_house_probabilities = compute_sigmoid(
                standardized_scores_with_intercept.dot(current_house_coefficients)
            )
            coefficient_gradient = (
                1 / student_count
            ) * standardized_scores_with_intercept.T.dot(
                predicted_house_probabilities - is_student_in_current_house
            )
            current_house_coefficients -= learning_rate * coefficient_gradient

        house_coefficient_table[int(current_house_code), :] = current_house_coefficients

    return house_coefficient_table


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            training_discipline_score_table,
            target_house_codes,
            house_code_by_name,
            house_name_by_code,
            discipline_score_column_names,
        ) = load_and_prepare_training_dataset(cli_arguments.input_csv_path)

        (
            standardized_discipline_scores,
            average_score_by_discipline,
            standard_deviation_by_discipline,
        ) = standardize_discipline_scores(training_discipline_score_table)

        student_count = standardized_discipline_scores.shape[0]
        standardized_discipline_scores_with_intercept = np.hstack(
            [np.ones((student_count, 1)), standardized_discipline_scores]
        )

        house_coefficient_table = fit_one_vs_rest_house_classifier(
            standardized_discipline_scores_with_intercept,
            target_house_codes,
            cli_arguments.learning_rate,
            cli_arguments.iteration_count,
        )

        trained_parameter_bundle = {
            "thetas": house_coefficient_table.tolist(),
            "mu": average_score_by_discipline,
            "sigma": standard_deviation_by_discipline,
            "features": discipline_score_column_names,
            "house_map": house_code_by_name,
            "inv_house_map": house_name_by_code,
        }

        with open(cli_arguments.output_parameter_path, "w") as output_file:
            json.dump(trained_parameter_bundle, output_file)

        print(
            f"→ Poids et parametres enregistres dans {cli_arguments.output_parameter_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
