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
        - thetas : poids appris (maisons x disciplines+bias)
        - mu : moyenne des matieres utilisees pour normaliser
        - sigma : ecart-types utilises pour normaliser
        - features : noms des disciplines (cle JSON conservee pour compatibilite)
        - inv_house_map : mapping code maison -> maison

    Args:
        trained_parameter_file_path (str): chemin vers le fichier JSON

    Returns:
        house_discipline_weights (numpy.ndarray): poids appris
        average_score_by_discipline (list[float]): moyenne discipline par discipline
        standard_deviation_by_discipline (list[float]): ecart-type discipline par discipline
        house_name_by_code (dict[int, str]): nom des differentes maisons
        discipline_names (list[str]): noms des matieres
    """
    with open(trained_parameter_file_path, "r") as parameter_file:
        trained_parameter_bundle = json.load(parameter_file)

    house_discipline_weights = np.array(trained_parameter_bundle["thetas"])
    average_score_by_discipline = trained_parameter_bundle["mu"]
    standard_deviation_by_discipline = trained_parameter_bundle["sigma"]
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle["inv_house_map"].items()
    }
    discipline_names = trained_parameter_bundle["features"]

    return (
        house_discipline_weights,
        average_score_by_discipline,
        standard_deviation_by_discipline,
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
        student_index_list (list[int]): liste des index du dataset
        student_discipline_scores (pandas.DataFrame): disciplines dans le bon ordre
    """
    dataset = pd.read_csv(dataset_csv_path)

    if "Index" not in dataset.columns:
        raise ValueError("La colonne 'Index' est manquante dans le fichier CSV.")

    missing_discipline_names = [
        discipline_name for discipline_name in discipline_names
        if discipline_name not in dataset.columns
    ]
    if missing_discipline_names:
        raise ValueError(
            f"Disciplines manquantes dans le fichier CSV: {missing_discipline_names}"
        )

    student_index_list = dataset["Index"].tolist()
    student_discipline_scores = dataset[discipline_names].copy()

    return student_index_list, student_discipline_scores


def standardize_discipline_scores(
    student_discipline_scores,
    average_score_by_discipline,
    standard_deviation_by_discipline,
):
    """
    Normalise les notes du dataset avec les parametres du train.

    Args:
        student_discipline_scores (pandas.DataFrame): notes brutes du dataset
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

    completed_student_discipline_scores = student_discipline_scores.astype(float).copy()
    for discipline_index, discipline_name in enumerate(
        completed_student_discipline_scores.columns
    ):
        completed_student_discipline_scores[discipline_name] = (
            completed_student_discipline_scores[discipline_name].fillna(
                average_score_by_discipline[discipline_index]
            )
        )

    discipline_scores_by_student = completed_student_discipline_scores.to_numpy(
        dtype=float
    )
    standardized_discipline_scores = (
        discipline_scores_by_student - average_score_by_discipline
    ) / safe_standard_deviation_by_discipline
    return standardized_discipline_scores


def predict_house_names(
    student_discipline_scores_with_bias,
    house_discipline_weights,
    house_name_by_code,
):
    """
    Predit la maison de chaque eleve a partir des notes du dataset et des coefficients appris.

    Args:
        student_discipline_scores_with_bias (numpy.ndarray): notes normalisees + bias
        house_discipline_weights (numpy.ndarray): poids appris
        house_name_by_code (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    house_logit_scores = student_discipline_scores_with_bias.dot(
        house_discipline_weights.T
    )
    house_logit_scores = np.clip(house_logit_scores, -500, 500)
    house_probability_scores = 1 / (1 + np.exp(-house_logit_scores))
    predicted_house_codes = np.argmax(house_probability_scores, axis=1)
    predicted_house_names = [
        house_name_by_code[int(house_code)] for house_code in predicted_house_codes
    ]
    return predicted_house_names


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            house_discipline_weights,
            average_score_by_discipline,
            standard_deviation_by_discipline,
            house_name_by_code,
            discipline_names,
        ) = load_house_classifier_parameters(cli_arguments.trained_parameter_file_path)

        student_index_list, student_discipline_scores = load_observations(
            cli_arguments.dataset_csv_path,
            discipline_names
        )

        standardized_discipline_scores = standardize_discipline_scores(
            student_discipline_scores,
            average_score_by_discipline,
            standard_deviation_by_discipline
        )

        student_count = standardized_discipline_scores.shape[0]
        student_discipline_scores_with_bias = np.hstack(
            [np.ones((student_count, 1)), standardized_discipline_scores]
        )

        predicted_house_names = predict_house_names(
            student_discipline_scores_with_bias,
            house_discipline_weights,
            house_name_by_code
        )

        prediction_output = pd.DataFrame(
            {"Index": student_index_list, "Hogwarts House": predicted_house_names}
        )
        prediction_output.to_csv(cli_arguments.output_csv_path, index=False)
        print(
            f"→ Fichier de prediction enregistre dans {cli_arguments.output_csv_path}"
        )

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
