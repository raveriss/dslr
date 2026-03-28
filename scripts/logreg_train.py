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
        student_discipline_scores (pandas.DataFrame): notes numeriques par matiere, sans la colonne Hogwarts House
        assigned_house_codes_by_student (numpy.ndarray): codes maisons attendus (0-3)
        house_code_by_name (dict[str, int]): mapping maison -> code maison
        house_name_by_code (dict[int, str]): mapping code maison -> maison
        discipline_names (list[str]): noms des disciplines
    """
    raw_students_dataset = pd.read_csv(input_csv_path)
    discipline_names = get_discipline_names(raw_students_dataset)

    students_with_complete_discipline_scores = raw_students_dataset.dropna(
        subset=["Hogwarts House"] + discipline_names
    )

    student_discipline_scores = (
        students_with_complete_discipline_scores[discipline_names].reset_index(drop=True)
    )

    house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house_code_by_name = {house_name: code for code, house_name in enumerate(house_names)}
    house_name_by_code = {code: house_name for house_name, code in house_code_by_name.items()}
    assigned_house_codes_by_student = students_with_complete_discipline_scores["Hogwarts House"].map(
        house_code_by_name
    ).to_numpy(dtype=int)

    return (
        student_discipline_scores,
        assigned_house_codes_by_student,
        house_code_by_name,
        house_name_by_code,
        discipline_names,
    )


def standardize_discipline_scores(student_discipline_scores):
    """
    Centre et reduit les notes par matiere.
    Formule appliquee: (note_eleve_matiere - moyenne_matiere) / ecart_type_matiere.

    Returns:
        standardized_discipline_scores (numpy.ndarray): notes normalisees
        average_score_by_discipline (list[float]): moyenne de chaque matiere
        standard_deviation_by_discipline (list[float]): ecart-type de chaque matiere
    """
    discipline_scores_by_student = (
        student_discipline_scores.to_numpy(dtype=float)
    )
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
    student_discipline_scores_with_bias,
    assigned_house_codes_by_student,
    learning_rate,
    iteration_count,
):
    """
    Entraine un classifieur logistique one-vs-all.

    Args:
        student_discipline_scores_with_bias (numpy.ndarray): notes normalisees + colonne de bias
        assigned_house_codes_by_student (numpy.ndarray): codes maisons 0, 1, 2 ou 3
        learning_rate (float): taux d'apprentissage
        iteration_count (int): nombre d'iterations

    Returns:
        numpy.ndarray: poids des maisons par discipline, avec le bias en colonne 0
    """
    student_count, discipline_plus_bias_count = student_discipline_scores_with_bias.shape

    unique_house_codes = np.unique(assigned_house_codes_by_student)
    
    house_count = len(unique_house_codes)
    house_discipline_weights = np.zeros((house_count, discipline_plus_bias_count))

    # "Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for current_house_code in unique_house_codes:
        print("")
        if current_house_code == 0:
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            print(f"/*                                 HOUSE                                     */")
            print(f"/*                               GRYFFINDOR                                  */")
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        elif current_house_code == 1:
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            print(f"/*                                 HOUSE                                     */")
            print(f"/*                               HUFFLEPUFF                                  */")
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        elif current_house_code == 2:
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            print(f"/*                                 HOUSE                                     */")
            print(f"/*                               RAVENCLAW                                   */")
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        elif current_house_code == 3:
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            print(f"/*                                 HOUSE                                     */")
            print(f"/*                               SLYTHERIN                                   */")
            print(f"/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")

        current_house_weights = np.zeros(discipline_plus_bias_count)
        is_student_assigned_to_current_house = (
            assigned_house_codes_by_student == current_house_code
        ).astype(float)

        print(f"\nIS_STUDENT_ASSIGNED_TO_CURRENT_HOUSE : {is_student_assigned_to_current_house}")

        for _ in range(iteration_count):
            print("")
            print(f"                        /*   -'-,-'-,-'-,-'-,-'-,-   */")
            print(f"                        /*    ITERATION_COUNT : {_}    */")
            print(f"                        /*   -'-,-'-,-'-,-'-,-'-,-   */")

            predicted_probability_of_current_house = compute_sigmoid(
                student_discipline_scores_with_bias.dot(current_house_weights)
            )
            print(
                "\nPREDICTED_PROBABILITY_OF_CURRENT_HOUSE"
                "\nCALCULE :"
                "\ncompute_sigmoid(student_discipline_scores_with_bias.dot(current_house_weights))"
                f"\ncompute_sigmoid(student_discipline_scores_with_bias.dot({current_house_weights}))"
                f"\ncompute_sigmoid({student_discipline_scores_with_bias.dot(current_house_weights)})"
                "\n--------------------------------------------------------------"    
                f"\n= {compute_sigmoid(student_discipline_scores_with_bias.dot(current_house_weights))}"
            )
            print(f"\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            prediction_error_by_student = (
                predicted_probability_of_current_house
                - is_student_assigned_to_current_house
            )
            print("\nPREDICTION_ERROR_BY_STUDENT")
            print(f"CALCULE :\npredicted_probability_of_current_house - is_student_assigned_to_current_house")
            print(f"{predicted_probability_of_current_house} - {is_student_assigned_to_current_house}")
            print(
                "--------------------------------------------------------------"
                f"\n= {prediction_error_by_student}"
            )
            print(f"\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            bias_and_standardized_discipline_scores_error_sum = (
                student_discipline_scores_with_bias.T.dot(
                    prediction_error_by_student
                )
            )
            print(
                "\nBIAS_AND_STANDARDIZED_DISCIPLINE_SCORES_ERROR_SUM"
                "\nCALCULE :"
                "\nstudent_discipline_scores_with_bias.T.dot(prediction_error_by_student)"
                f"\n{student_discipline_scores_with_bias}"
                f"\n                                   .T.dot({prediction_error_by_student})"      
                "\n--------------------------------------------------------------"
                f"\n= {bias_and_standardized_discipline_scores_error_sum}"
            )
            print(f"\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            current_house_weight_gradient = (
                1 / student_count
            ) * bias_and_standardized_discipline_scores_error_sum
            print(
                "\nCURRENT_HOUSE_WEIGHT_GRADIENT"
                "\nCALCULE :\n(1 / student_count) * bias_and_standardized_discipline_scores_error_sum"
                f"\n(1 / {student_count}) * {bias_and_standardized_discipline_scores_error_sum}"
                f"\n({1 / student_count}) * {bias_and_standardized_discipline_scores_error_sum}"
                "\n--------------------------------------------------------------"
                f"\n= {current_house_weight_gradient}"
            )
            print(f"\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
            print(
                "\nCURRENT_HOUSE_WEIGHTS"
                "\nCALCULE :\ncurrent_house_weights - learning_rate * current_house_weight_gradient"
                f"\n{current_house_weights} - {learning_rate} * {current_house_weight_gradient}"
                f"\n{current_house_weights} - {learning_rate * current_house_weight_gradient}"

                "\n--------------------------------------------------------------"
            )
            current_house_weights -= learning_rate * current_house_weight_gradient
            print(f"{current_house_weights}")

            
        house_discipline_weights[int(current_house_code), :] = current_house_weights
        print(f"\nHOUSE_DISCIPLINE_WEIGHTS : \n{house_discipline_weights}")
    return house_discipline_weights


def main():
    try:
        cli_arguments = parse_command_line_arguments()
        (
            student_discipline_scores,
            assigned_house_codes_by_student,
            house_code_by_name,
            house_name_by_code,
            discipline_names,
        ) = load_and_prepare_dataset(cli_arguments.input_csv_path)
        print(f"\nstudent_discipline_scores : \n{student_discipline_scores}")

        (
            standardized_discipline_scores,
            average_score_by_discipline,
            standard_deviation_by_discipline,
        ) = standardize_discipline_scores(student_discipline_scores)

        student_count = standardized_discipline_scores.shape[0]
        student_discipline_scores_with_bias = np.hstack(
            [np.ones((student_count, 1)), standardized_discipline_scores]
        )
        print(f"\nstandardized_discipline_scores : \n{standardized_discipline_scores}")
        house_discipline_weights = fit_one_vs_rest_house_classifier(
            student_discipline_scores_with_bias,
            assigned_house_codes_by_student,
            cli_arguments.learning_rate,
            cli_arguments.iteration_count,
        )

        trained_parameter_bundle = {
            "thetas": house_discipline_weights.tolist(),
            "mu": average_score_by_discipline,
            "sigma": standard_deviation_by_discipline,
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
