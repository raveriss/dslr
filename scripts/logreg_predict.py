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
          - trained_parameter_file_path (str): fichier json contenant les parametres appris
          - output_csv_path (str): chemin du fichier de sortie a genere
    """
    argument_parser = argparse.ArgumentParser(
        description="Predit la maison de chaque eleve a partir d'un dataset test et d'un fichier de parametres"
    )
    argument_parser.add_argument(
        "input_csv_path",
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


def load_house_classifier_components(trained_parameter_file_path):
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
        subject_score_column_averages (list[float]): moyenne colonne par colonne
        subject_score_column_std (list[float]): ecart-type colonne par colonne
        house_name_by_code (dict[int, str]): nom des differentes maisons
        subject_score_column_names (list[str]): noms des matieres
    """
    with open(trained_parameter_file_path, "r") as parameter_file:
        trained_parameter_bundle = json.load(parameter_file)

    house_coefficient_table = np.array(trained_parameter_bundle["thetas"])
    subject_score_column_averages = trained_parameter_bundle["mu"]
    subject_score_column_std = trained_parameter_bundle["sigma"]
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle["inv_house_map"].items()
    }
    subject_score_column_names = trained_parameter_bundle["features"]

    return (
        house_coefficient_table,
        subject_score_column_averages,
        subject_score_column_std,
        house_name_by_code,
        subject_score_column_names,
    )


def get_numeric_subject_score_columns(dataset_table):
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


def load_test_observation_inputs(input_csv_path, subject_score_column_names):
    """
    Lit le fichier dataset_test.csv.

    Args:
        input_csv_path (str): chemin du fichier dataset_test.csv
        subject_score_column_names (list[str]): colonnes attendues depuis weights.json

    Returns:
        student_index_list (list[int]): liste des index du dataset test
        test_subject_score_table (pandas.DataFrame): colonnes de matieres dans le bon ordre
    """
    test_dataset_table = pd.read_csv(input_csv_path)

    if "Index" not in test_dataset_table.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier test.")

    missing_subject_score_columns = [
        subject_column_name for subject_column_name in subject_score_column_names
        if subject_column_name not in test_dataset_table.columns
    ]
    if missing_subject_score_columns:
        raise ValueError(
            f"Colonnes manquantes dans le fichier test: {missing_subject_score_columns}"
        )

    student_index_list = test_dataset_table["Index"].tolist()
    test_subject_score_table = test_dataset_table[subject_score_column_names].copy()

    return student_index_list, test_subject_score_table


def standardize_test_subject_scores(
    test_subject_score_table,
    subject_score_column_averages,
    subject_score_column_std,
):
    """
    Normalise les notes de test avec les parametres du train.

    Args:
        test_subject_score_table (pandas.DataFrame): notes brutes du dataset test
        subject_score_column_averages (list[float]): moyenne des matieres calculee au train
        subject_score_column_std (list[float]): ecart-type calcule au train

    Returns:
        numpy.ndarray: tableau normalise
    """
    subject_score_column_averages = np.array(subject_score_column_averages, dtype=float)
    subject_score_column_std = np.array(subject_score_column_std, dtype=float)
    safe_subject_score_column_std = np.where(
        subject_score_column_std == 0,
        1.0,
        subject_score_column_std,
    )

    completed_test_subject_score_table = test_subject_score_table.astype(float).copy()
    for subject_column_index, column_name in enumerate(completed_test_subject_score_table.columns):
        completed_test_subject_score_table[column_name] = (
            completed_test_subject_score_table[column_name].fillna(
                subject_score_column_averages[subject_column_index]
            )
        )

    test_subject_score_array = completed_test_subject_score_table.to_numpy(dtype=float)
    standardized_test_subject_score_array = (
        test_subject_score_array - subject_score_column_averages
    ) / safe_subject_score_column_std
    return standardized_test_subject_score_array


def predict_house_names(
    standardized_scores_with_intercept,
    house_coefficient_table,
    house_name_by_code,
):
    """
    Predit la maison de chaque eleve a partir des notes test et des coefficients appris.

    Args:
        standardized_scores_with_intercept (numpy.ndarray): notes test normalisees + interception
        house_coefficient_table (numpy.ndarray): tableau des coefficients appris
        house_name_by_code (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    house_logit_table = standardized_scores_with_intercept.dot(house_coefficient_table.T)
    house_logit_table = np.clip(house_logit_table, -500, 500)
    house_probability_table = 1 / (1 + np.exp(-house_logit_table))
    predicted_house_code_array = np.argmax(house_probability_table, axis=1)
    predicted_house_names = [
        house_name_by_code[int(house_code)] for house_code in predicted_house_code_array
    ]
    return predicted_house_names


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            house_coefficient_table,
            subject_score_column_averages,
            subject_score_column_std,
            house_name_by_code,
            subject_score_column_names,
        ) = load_house_classifier_components(cli_arguments.trained_parameter_file_path)

        student_index_list, test_subject_score_table = load_test_observation_inputs(
            cli_arguments.input_csv_path,
            subject_score_column_names
        )

        standardized_test_subject_score_array = standardize_test_subject_scores(
            test_subject_score_table,
            subject_score_column_averages,
            subject_score_column_std
        )

        student_count = standardized_test_subject_score_array.shape[0]
        standardized_scores_with_intercept = np.hstack(
            [np.ones((student_count, 1)), standardized_test_subject_score_array]
        )

        predicted_house_names = predict_house_names(
            standardized_scores_with_intercept,
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
