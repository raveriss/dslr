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
          - trained_parameter_json_file_path (str): fichier json contenant les parametres appris
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
        "trained_parameter_json_file_path",
        help="Fichier JSON contenant les parametres appris (creer par logreg_train.py)"
    )
    argument_parser.add_argument(
        "--out", "-o",
        dest="output_csv_path",
        default="houses.csv",
        help="Fichier de sortie contenant les predictions"
    )
    return argument_parser.parse_args()


def load_house_classifier_parameters(trained_parameter_json_file_path):
    """
    Charge le fichier JSON contenant :
        - thetas : poids appris (maisons x disciplines+bias)
        - mu : moyenne des matieres utilisees pour normaliser
        - sigma : ecart-types utilises pour normaliser
        - features : noms des disciplines (cle JSON conservee pour compatibilite)
        - inv_house_map : mapping code maison -> maison

    Args:
        trained_parameter_json_file_path (str): chemin vers le fichier JSON

    Returns:
        house_discipline_weights_with_bias (numpy.ndarray): poids appris (disciplines+bias)
        average_discipline_scores (list[float]): moyenne discipline par discipline
        discipline_standard_deviations (list[float]): ecart-type discipline par discipline
        house_name_by_code (dict[int, str]): nom des differentes maisons
        discipline_names (list[str]): noms des matieres
    """
    with open(trained_parameter_json_file_path, "r") as parameter_file:
        trained_parameter_bundle = json.load(parameter_file)

    house_discipline_weights_with_bias = np.array(trained_parameter_bundle["thetas"])
    average_discipline_scores = trained_parameter_bundle["mu"]
    discipline_standard_deviations = trained_parameter_bundle["sigma"]
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle["inv_house_map"].items()
    }
    discipline_names = trained_parameter_bundle["features"]


    return (
        house_discipline_weights_with_bias,
        average_discipline_scores,
        discipline_standard_deviations,
        house_name_by_code,
        discipline_names,
    )


def get_discipline_names(dataset):
    """
    Identifie et retourne les noms des disciplines numeriques.

    Args:
        dataset (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]: noms de disciplines, excluant 'Index' et 'Hogwarts House'.
    """
    discipline_names = []
    for discipline_name in dataset.columns:
        discipline_kind = dataset[discipline_name].dtype.kind
        if discipline_kind in ("i", "f"):
            discipline_names.append(discipline_name)

    if "Index" in discipline_names:
        discipline_names.remove("Index")
    if "Hogwarts House" in discipline_names:
        discipline_names.remove("Hogwarts House")

    return discipline_names


def load_observations(dataset_csv_path, discipline_names):
    """
    Lit le fichier CSV des observations.

    Args:
        dataset_csv_path (str): chemin du fichier CSV
        discipline_names (list[str]): disciplines attendues depuis weights.json

    Returns:
        index_list_of_students (list[int]): liste des index du dataset
        students_discipline_scores (pandas.DataFrame): disciplines dans le bon ordre
    """
    raw_students_dataset = pd.read_csv(dataset_csv_path)

    if "Index" not in raw_students_dataset.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier CSV.")

    missing_discipline_names = [
        discipline_name for discipline_name in discipline_names
        if discipline_name not in raw_students_dataset.columns
    ]
    if missing_discipline_names:
        raise ValueError(
            f"Disciplines manquantes dans le fichier CSV: {missing_discipline_names}"
        )

    index_list_of_students = raw_students_dataset["Index"].tolist()
    students_discipline_scores = raw_students_dataset[discipline_names].copy()

    return index_list_of_students, students_discipline_scores


def standardize_discipline_scores(
    students_discipline_scores,
    average_discipline_scores,
    discipline_standard_deviations,
):
    """
    Normalise les notes du dataset avec les parametres du train.

    Args:
        students_discipline_scores (pandas.DataFrame): notes brutes du dataset
        average_discipline_scores (list[float]): moyenne des matieres calculee au train
        discipline_standard_deviations (list[float]): ecart-type calcule au train

    Returns:
        numpy.ndarray: tableau normalise
    """

    average_discipline_scores = np.array(average_discipline_scores, dtype=float)
    discipline_standard_deviations = np.array(discipline_standard_deviations, dtype=float)

    discipline_standard_deviations_without_zero = np.where(
        discipline_standard_deviations == 0,
        1.0,
        discipline_standard_deviations,
    )

    students_discipline_scores_with_missing_scores = students_discipline_scores.astype(float).copy()

    for discipline_index, discipline_name in enumerate(
        students_discipline_scores_with_missing_scores.columns
    ):
        students_discipline_scores_with_missing_scores[discipline_name] = (
            students_discipline_scores_with_missing_scores[discipline_name].fillna(
                average_discipline_scores[discipline_index]
            )
        )

    discipline_scores_by_student = students_discipline_scores_with_missing_scores.to_numpy(
        dtype=float
    )

    standardized_students_discipline_scores = (
        discipline_scores_by_student - average_discipline_scores
    ) / discipline_standard_deviations_without_zero
    return standardized_students_discipline_scores


def predict_house_names(
    students_discipline_scores_with_bias,
    house_discipline_weights_with_bias,
    house_name_by_code,
):
    """
    Predit la maison de chaque eleve a partir des notes du dataset et des coefficients appris.

    Args:
        students_discipline_scores_with_bias (numpy.ndarray): notes normalisees + bias
        house_discipline_weights_with_bias (numpy.ndarray): poids appris (disciplines+bias)
        house_name_by_code (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    house_scores_before_value_limit_for_all_students = students_discipline_scores_with_bias.dot(
        house_discipline_weights_with_bias.T
    )
    print(f"house_scores_before_value_limit_for_all_students = {house_scores_before_value_limit_for_all_students}")

    house_scores_after_value_limit_for_all_students = np.clip(
        house_scores_before_value_limit_for_all_students, -500, 500
    )
    print(f"house_scores_after_value_limit_for_all_students = {house_scores_after_value_limit_for_all_students}")

    house_probability_scores_for_all_students = 1 / (1 + np.exp(
        -house_scores_after_value_limit_for_all_students
    ))
    print(f"house_probability_scores_for_all_students = {house_probability_scores_for_all_students}")
    predicted_house_codes_for_all_students = np.argmax(house_probability_scores_for_all_students, axis=1)
    print(f"predicted_house_codes_for_all_students = {predicted_house_codes_for_all_students}")

    predicted_house_names_for_all_students = [
        house_name_by_code[int(house_code)] for house_code in predicted_house_codes_for_all_students
    ]
    print(f"predicted_house_names_for_all_students = {predicted_house_names_for_all_students}")
    print(f"predicted_house_names_for_all_students = {len(predicted_house_names_for_all_students)}")

    return predicted_house_names_for_all_students


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            house_discipline_weights_with_bias,
            average_discipline_scores,
            discipline_standard_deviations,
            house_name_by_code,
            discipline_names,
        ) = load_house_classifier_parameters(cli_arguments.trained_parameter_json_file_path)

        index_list_of_students, students_discipline_scores = load_observations(
            cli_arguments.dataset_csv_path,
            discipline_names
        )

        standardized_students_discipline_scores = standardize_discipline_scores(
            students_discipline_scores,
            average_discipline_scores,
            discipline_standard_deviations
        )

        student_count = standardized_students_discipline_scores.shape[0]
        students_discipline_scores_with_bias = np.hstack(
            [np.ones((student_count, 1)), standardized_students_discipline_scores]
        )

        predicted_house_names_for_all_students = predict_house_names(
            students_discipline_scores_with_bias,
            house_discipline_weights_with_bias,
            house_name_by_code
        )

        prediction_output = pd.DataFrame(
            {"Index": index_list_of_students, "Hogwarts House": predicted_house_names_for_all_students}
        )
        prediction_output.to_csv(cli_arguments.output_csv_path, index=False)
        print(
            f"→ Fichier de prediction enregistre dans {cli_arguments.output_csv_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
