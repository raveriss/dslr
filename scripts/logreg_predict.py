import argparse
import json

import numpy as np
import pandas as pd


def parse_command_line_arguments():
    """
    Parse les arguments de la ligne de commande pour logreg_predict.py.

    Returns:
        Namespace: contient les attributs suivants
          - dataset_csv_path (str): chemin vers le CSV des observations (ex: dataset_test.csv)
          - trained_parameter_file_path (str): fichier json contenant les parametres appris
          - output_csv_path (str): chemin du fichier de sortie a genere
    """
    argument_parser = argparse.ArgumentParser(
        description="Predit la maison de chaque eleve a partir d'un dataset et d'un fichier de parametres"
    )
    argument_parser.add_argument(
        "dataset_csv_path",
        help="Chemin vers le fichier dataset_test.csv (sans colonne House)"
    )
    argument_parser.add_argument(
        "trained_parameter_file_path",
        help="Fichier JSON contenant les parametres appris (creer par logreg_train.py)"
    )
    argument_parser.add_argument(
        "--out", "-o",
        dest="output_csv_path",
        default="houses.csv",
        help="Fichier de sortie contenant les predictions"
    )
    return argument_parser.parse_args()


def load_house_classifier_parameters(trained_parameter_file_path):
    """
    Charge le fichier JSON contenant :
        - thetas : tableau des coefficients (K x n+1)
        - mu : moyenne des matieres utilisees pour normaliser
        - sigma : ecart-types utilises pour normaliser
        - features : nom de toutes les matieres
        - inv_house_map : mapping code maison -> maison

    Args:
        trained_parameter_file_path (str): chemin vers le fichier JSON

    Returns:
        house_coefficient_table (numpy.ndarray): coefficients appris
        average_score_by_discipline (list[float]): moyenne colonne par colonne
        standard_deviation_by_discipline (list[float]): ecart-type colonne par colonne
        house_name_by_code (dict[int, str]): nom des differentes maisons
        discipline_score_column_names (list[str]): noms des matieres
    """
    with open(trained_parameter_file_path, "r") as parameter_file:
        trained_parameter_bundle = json.load(parameter_file)

    house_coefficient_table = np.array(trained_parameter_bundle["thetas"])
    average_score_by_discipline = trained_parameter_bundle["mu"]
    standard_deviation_by_discipline = trained_parameter_bundle["sigma"]
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle["inv_house_map"].items()
    }
    discipline_score_column_names = trained_parameter_bundle["features"]

    return (
        house_coefficient_table,
        average_score_by_discipline,
        standard_deviation_by_discipline,
        house_name_by_code,
        discipline_score_column_names,
    )


def get_numeric_discipline_score_columns(dataset_table):
    """
    Identifie et retourne les colonnes numeriques correspondant aux matieres.

    Args:
        dataset_table (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de colonnes numeriques, excluant 'Index' et 'Hogwarts House'.
    """
    numeric_column_names = []
    for column_name in dataset_table.columns:
        column_kind = dataset_table[column_name].dtype.kind
        if column_kind in ("i", "f"):
            numeric_column_names.append(column_name)

    if "Index" in numeric_column_names:
        numeric_column_names.remove("Index")
    if "Hogwarts House" in numeric_column_names:
        numeric_column_names.remove("Hogwarts House")

    return numeric_column_names


def load_observations(dataset_csv_path, discipline_score_column_names):
    """
    Lit le fichier CSV des observations.

    Args:
        dataset_csv_path (str): chemin du fichier CSV
        discipline_score_column_names (list[str]): colonnes attendues depuis weights.json

    Returns:
        student_index_list (list[int]): liste des index du dataset
        discipline_score_table (pandas.DataFrame): colonnes de matieres dans le bon ordre
    """
    dataset_table = pd.read_csv(dataset_csv_path)

    if "Index" not in dataset_table.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier CSV.")

    missing_discipline_score_columns = [
        discipline_column_name for discipline_column_name in discipline_score_column_names
        if discipline_column_name not in dataset_table.columns
    ]
    if missing_discipline_score_columns:
        raise ValueError(
            f"Colonnes manquantes dans le fichier CSV: {missing_discipline_score_columns}"
        )

    student_index_list = dataset_table["Index"].tolist()
    discipline_score_table = dataset_table[discipline_score_column_names].copy()

    return student_index_list, discipline_score_table


def standardize_discipline_scores(
    discipline_score_table,
    average_score_by_discipline,
    standard_deviation_by_discipline,
):
    """
    Normalise les notes du dataset avec les parametres du train.

    Args:
        discipline_score_table (pandas.DataFrame): notes brutes du dataset
        average_score_by_discipline (list[float]): moyenne des matieres calculee au train
        standard_deviation_by_discipline (list[float]): ecart-type calcule au train

    Returns:
        numpy.ndarray: tableau normalise
    """
    average_score_by_discipline = np.array(average_score_by_discipline, dtype=float)
    standard_deviation_by_discipline = np.array(standard_deviation_by_discipline, dtype=float)
    safe_standard_deviation_by_discipline = np.where(
        standard_deviation_by_discipline == 0,
        1.0,
        standard_deviation_by_discipline,
    )

    completed_discipline_score_table = discipline_score_table.astype(float).copy()
    for discipline_column_index, column_name in enumerate(
        completed_discipline_score_table.columns
    ):
        completed_discipline_score_table[column_name] = (
            completed_discipline_score_table[column_name].fillna(
                average_score_by_discipline[discipline_column_index]
            )
        )

    discipline_scores_by_student = completed_discipline_score_table.to_numpy(
        dtype=float
    )
    standardized_discipline_scores = (
        discipline_scores_by_student - average_score_by_discipline
    ) / safe_standard_deviation_by_discipline
    return standardized_discipline_scores


def predict_house_names(
    standardized_discipline_scores_with_intercept,
    house_coefficient_table,
    house_name_by_code,
):
    """
    Predit la maison de chaque eleve a partir des notes du dataset et des coefficients appris.

    Args:
        standardized_discipline_scores_with_intercept (numpy.ndarray): notes normalisees + interception
        house_coefficient_table (numpy.ndarray): tableau des coefficients appris
        house_name_by_code (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    house_logit_table = standardized_discipline_scores_with_intercept.dot(
        house_coefficient_table.T
    )
    house_logit_table = np.clip(house_logit_table, -500, 500)
    house_probability_table = 1 / (1 + np.exp(-house_logit_table))
    predicted_house_codes = np.argmax(house_probability_table, axis=1)
    predicted_house_names = [
        house_name_by_code[int(house_code)] for house_code in predicted_house_codes
    ]
    return predicted_house_names


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            house_coefficient_table,
            average_score_by_discipline,
            standard_deviation_by_discipline,
            house_name_by_code,
            discipline_score_column_names,
        ) = load_house_classifier_parameters(cli_arguments.trained_parameter_file_path)

        student_index_list, discipline_score_table = load_observations(
            cli_arguments.dataset_csv_path,
            discipline_score_column_names
        )

        standardized_discipline_scores = standardize_discipline_scores(
            discipline_score_table,
            average_score_by_discipline,
            standard_deviation_by_discipline
        )

        student_count = standardized_discipline_scores.shape[0]
        standardized_discipline_scores_with_intercept = np.hstack(
            [np.ones((student_count, 1)), standardized_discipline_scores]
        )

        predicted_house_names = predict_house_names(
            standardized_discipline_scores_with_intercept,
            house_coefficient_table,
            house_name_by_code
        )

        prediction_output_table = pd.DataFrame(
            {"Index": student_index_list, "Hogwarts House": predicted_house_names}
        )
        prediction_output_table.to_csv(cli_arguments.output_csv_path, index=False)
        print(
            f"→ Fichier de prediction enregistre dans {cli_arguments.output_csv_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
