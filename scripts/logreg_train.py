import argparse
import json

import numpy as np
import pandas as pd

try:
    from analysis_log import AnalysisLogger
except ImportError:
    from scripts.analysis_log import AnalysisLogger


def parse_command_line_arguments():
    """
    Parse les arguments de la ligne de commande pour logreg_train.py.

    Returns:
        Namespace: contient les attributs suivants
          - input_csv_path (str): chemin vers dataset_train.csv
          - learning_rate (float): taux d'apprentissage
          - iteration_count (int): nombre d'iterations de gradient descent
          - output_parameter_path (str): chemin du fichier de sortie des parametres appris
          - enable_analysis_log (bool): active les logs detailles d'analyse
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
    argument_parser.add_argument(
        "--analysis-log",
        dest="enable_analysis_log",
        action="store_true",
        help="Active les logs detailles pour analyser l'entrainement."
    )
    return argument_parser.parse_args()


def get_discipline_names(dataset):
    """
    Identifie et retourne les noms des disciplines numeriques.

    Args:
        dataset (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de disciplines, excluant 'Index'.
    """
    discipline_names = []
    for discipline_name in dataset.columns:
        discipline_kind = dataset[discipline_name].dtype.kind
        if discipline_kind in ("i", "f"):
            discipline_names.append(discipline_name)

    if "Index" in discipline_names:
        discipline_names.remove("Index")

    return discipline_names


def log_discipline_column_diagnostics(raw_students_dataset, discipline_names):
    """
    Affiche des logs utiles pour comprendre pourquoi certaines disciplines sont ignorees.
    """

    candidate_discipline_names = [
        "Arithmancy",
        "Astronomy",
        "Muggle Studies",
        "Potions",
        "Flying",
    ]

    for discipline_name in candidate_discipline_names:
        if discipline_name not in raw_students_dataset.columns:
            print(f"- {discipline_name}: colonne absente du CSV")
            continue

        discipline_scores = raw_students_dataset[discipline_name]
        data_type_name = str(discipline_scores.dtype)
        nan_count = int(discipline_scores.isna().sum())
        sample_values = discipline_scores.head(5).tolist()

    ignored_discipline_names = [
        discipline_name for discipline_name in candidate_discipline_names
        if discipline_name in raw_students_dataset.columns and discipline_name not in discipline_names
    ]


def load_and_prepare_dataset(input_csv_path):
    """
    Lit le CSV d'entrainement et prepare X, y pour l'entrainement.

    Returns:
        students_disciplines_scores (pandas.DataFrame): notes numeriques par matiere, sans la colonne Hogwarts House
        assigned_house_codes_for_students (numpy.ndarray): codes maisons attendus (0-3)
        house_code_by_name (dict[str, int]): mapping maison -> code maison
        house_name_by_code (dict[int, str]): mapping code maison -> maison
        discipline_names (list[str]): noms des disciplines
    """
    raw_students_dataset = pd.read_csv(input_csv_path)
    discipline_names = get_discipline_names(raw_students_dataset)

    students_with_complete_disciplines_scores = raw_students_dataset.dropna(
        subset=["Hogwarts House"] + discipline_names
    )

    students_disciplines_scores = (
        students_with_complete_disciplines_scores[discipline_names].reset_index(drop=True)
    )

    house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house_code_by_name = {house_name: code for code, house_name in enumerate(house_names)}
    house_name_by_code = {code: house_name for house_name, code in house_code_by_name.items()}
    assigned_house_codes_for_students = students_with_complete_disciplines_scores["Hogwarts House"].map(
        house_code_by_name
    ).to_numpy(dtype=int)

    return (
        students_disciplines_scores,
        assigned_house_codes_for_students,
        house_code_by_name,
        house_name_by_code,
        discipline_names,
    )


def standardize_disciplines_scores(students_disciplines_scores):
    """
    Centre et reduit les notes par matiere.
    Formule appliquee: (note_eleve_matiere - moyenne_matiere) / ecart_type_matiere.

    Returns:
        standardized_disciplines_scores (numpy.ndarray): notes normalisees
        average_scores_by_disciplines (list[float]): moyenne de chaque matiere
        standard_deviations_by_disciplines (list[float]): ecart-type de chaque matiere
    """
    disciplines_scores_for_students = (
        students_disciplines_scores.to_numpy(dtype=float)
    )
    average_scores_by_disciplines = disciplines_scores_for_students.mean(axis=0)
    standard_deviations_by_disciplines = disciplines_scores_for_students.std(axis=0, ddof=0)
    standardized_disciplines_scores = (
        disciplines_scores_for_students - average_scores_by_disciplines
    ) / standard_deviations_by_disciplines

    return (
        standardized_disciplines_scores,
        average_scores_by_disciplines.tolist(),
        standard_deviations_by_disciplines.tolist(),
    )


def compute_sigmoid(linear_score_array):
    """Fonction sigmoide."""
    return 1.0 / (1.0 + np.exp(-linear_score_array))


def fit_one_vs_rest_house_classifier(
    students_disciplines_scores_with_bias,
    assigned_house_codes_for_students,
    learning_rate,
    iteration_count,
    analysis_logger,
):
    """
    Entraine un classifieur logistique one-vs-all.

    Args:
        students_disciplines_scores_with_bias (numpy.ndarray): notes normalisees + colonne de bias
        assigned_house_codes_for_students (numpy.ndarray): codes maisons 0, 1, 2 ou 3
        learning_rate (float): taux d'apprentissage
        iteration_count (int): nombre d'iterations

    Returns:
        numpy.ndarray: poids des maisons par disciplines, avec le bias en colonne 0
    """
    students_count, disciplines_plus_bias_count = students_disciplines_scores_with_bias.shape

    unique_house_codes = np.unique(assigned_house_codes_for_students)
    
    house_count = len(unique_house_codes)
    house_disciplines_weights = np.zeros((house_count, disciplines_plus_bias_count))

    for current_house_code in unique_house_codes:
        analysis_logger.log_house_header(current_house_code)

        current_house_weights = np.zeros(disciplines_plus_bias_count)
        are_students_assigned_to_current_house = (
            assigned_house_codes_for_students == current_house_code
        ).astype(float)
        analysis_logger.log_students_assigned_to_current_house(
            are_students_assigned_to_current_house
        )

        for _ in range(iteration_count):
            analysis_logger.log_iteration_header(_)

            predicted_probability_of_current_house = compute_sigmoid(
                students_disciplines_scores_with_bias.dot(current_house_weights)
            )
            analysis_logger.log_predicted_probability(
                students_disciplines_scores_with_bias,
                current_house_weights,
                predicted_probability_of_current_house,
            )
            prediction_error_by_students = (
                predicted_probability_of_current_house
                - are_students_assigned_to_current_house
            )
            analysis_logger.log_prediction_error(
                predicted_probability_of_current_house,
                are_students_assigned_to_current_house,
                prediction_error_by_students,
            )
            bias_and_standardized_disciplines_scores_error_sum = (
                students_disciplines_scores_with_bias.T.dot(
                    prediction_error_by_students
                )
            )
            analysis_logger.log_bias_and_standardized_disciplines_scores_error_sum(
                students_disciplines_scores_with_bias,
                prediction_error_by_students,
                bias_and_standardized_disciplines_scores_error_sum,
            )
            current_house_weight_gradient = (
                1 / students_count
            ) * bias_and_standardized_disciplines_scores_error_sum
            analysis_logger.log_current_house_weight_gradient(
                students_count,
                bias_and_standardized_disciplines_scores_error_sum,
                current_house_weight_gradient,
            )
            analysis_logger.log_current_house_weights_before_update(
                current_house_weights,
                learning_rate,
                current_house_weight_gradient,
            )
            current_house_weights -= learning_rate * current_house_weight_gradient
            analysis_logger.log_current_house_weights_after_update(current_house_weights)

        house_disciplines_weights[int(current_house_code), :] = current_house_weights
        analysis_logger.log_house_disciplines_weights(house_disciplines_weights)
    return house_disciplines_weights


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        analysis_logger = AnalysisLogger(cli_arguments.enable_analysis_log)
        (
            students_disciplines_scores,
            assigned_house_codes_for_students,
            house_code_by_name,
            house_name_by_code,
            discipline_names,
        ) = load_and_prepare_dataset(cli_arguments.input_csv_path)

        (
            standardized_disciplines_scores,
            average_scores_by_disciplines,
            standard_deviations_by_disciplines,
        ) = standardize_disciplines_scores(students_disciplines_scores)

        students_count = standardized_disciplines_scores.shape[0]
        students_disciplines_scores_with_bias = np.hstack(
            [np.ones((students_count, 1)), standardized_disciplines_scores]
        )
        analysis_logger.log_initial_scores(
            students_disciplines_scores,
            standardized_disciplines_scores,
        )
        house_disciplines_weights = fit_one_vs_rest_house_classifier(
            students_disciplines_scores_with_bias,
            assigned_house_codes_for_students,
            cli_arguments.learning_rate,
            cli_arguments.iteration_count,
            analysis_logger,
        )

        trained_parameter_bundle = {
            "thetas": house_disciplines_weights.tolist(),
            "mu": average_scores_by_disciplines,
            "sigma": standard_deviations_by_disciplines,
            "features": discipline_names,
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
