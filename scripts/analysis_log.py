class AnalysisLogger:
    """Logger verbeux pour analyser l'entrainement one-vs-all."""

    HOUSE_NAME_BY_CODE = {
        0: "GRYFFINDOR",
        1: "HUFFLEPUFF",
        2: "RAVENCLAW",
        3: "SLYTHERIN",
    }

    def __init__(self, enabled=False):
        self.enabled = enabled

    def log_initial_scores(self, students_disciplines_scores, standardized_disciplines_scores):
        if not self.enabled:
            return
        print(f"\nstudents_disciplines_scores : \n{students_disciplines_scores}")
        print(f"\nstandardized_disciplines_scores : \n{standardized_disciplines_scores}")

    def log_house_header(self, current_house_code):
        if not self.enabled:
            return
        house_name = self.HOUSE_NAME_BY_CODE.get(int(current_house_code), f"CODE_{current_house_code}")
        print("")
        print("/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        print("/*                                 HOUSE                                     */")
        print(f"/*                               {house_name:<44}*/")
        print("/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")

    def log_students_assigned_to_current_house(self, are_students_assigned_to_current_house):
        if not self.enabled:
            return
        print(f"\nARE_STUDENTS_ASSIGNED_TO_CURRENT_HOUSE : {are_students_assigned_to_current_house}")

    def log_iteration_header(self, iteration_index):
        if not self.enabled:
            return
        print("")
        print("                        /*   -'-,-'-,-'-,-'-,-'-,-   */")
        print(f"                        /*    ITERATION_COUNT : {iteration_index}    */")
        print("                        /*   -'-,-'-,-'-,-'-,-'-,-   */")

    def log_predicted_probability(
        self,
        students_disciplines_scores_with_bias,
        current_house_weights,
        predicted_probability_of_current_house,
    ):
        if not self.enabled:
            return
        linear_scores = students_disciplines_scores_with_bias.dot(current_house_weights)
        print(
            "\nPREDICTED_PROBABILITY_OF_CURRENT_HOUSE"
            "\nCALCULE :"
            "\ncompute_sigmoid(students_disciplines_scores_with_bias.dot(current_house_weights))"
            f"\ncompute_sigmoid(students_disciplines_scores_with_bias.dot({current_house_weights}))"
            f"\ncompute_sigmoid({linear_scores})"
            "\n--------------------------------------------------------------"
            f"\n= {predicted_probability_of_current_house}"
        )

    def log_prediction_error(
        self,
        predicted_probability_of_current_house,
        are_students_assigned_to_current_house,
        prediction_error_by_students,
    ):
        if not self.enabled:
            return
        print("\nPREDICTION_ERROR_BY_STUDENTS")
        print("CALCULE :\npredicted_probability_of_current_house - are_students_assigned_to_current_house")
        print(f"{predicted_probability_of_current_house} - {are_students_assigned_to_current_house}")
        print("--------------------------------------------------------------")
        print(f"= {prediction_error_by_students}")

    def log_bias_and_standardized_disciplines_scores_error_sum(
        self,
        students_disciplines_scores_with_bias,
        prediction_error_by_students,
        bias_and_standardized_disciplines_scores_error_sum,
    ):
        if not self.enabled:
            return
        print("\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        print(
            "\nBIAS_AND_STANDARDIZED_DISCIPLINES_SCORES_ERROR_SUM"
            "\nCALCULE :"
            "\nstudents_disciplines_scores_with_bias.T.dot(prediction_error_by_students)"
            f"\n{students_disciplines_scores_with_bias}"
            f"\n                                   .T.dot({prediction_error_by_students})"
            "\n--------------------------------------------------------------"
            f"\n= {bias_and_standardized_disciplines_scores_error_sum}"
        )

    def log_current_house_weight_gradient(
        self,
        students_count,
        bias_and_standardized_disciplines_scores_error_sum,
        current_house_weight_gradient,
    ):
        if not self.enabled:
            return
        print("\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        print(
            "\nCURRENT_HOUSE_WEIGHT_GRADIENT"
            "\nCALCULE :\n(1 / students_count) * bias_and_standardized_disciplines_scores_error_sum"
            f"\n(1 / {students_count}) * {bias_and_standardized_disciplines_scores_error_sum}"
            f"\n({1 / students_count}) * {bias_and_standardized_disciplines_scores_error_sum}"
            "\n--------------------------------------------------------------"
            f"\n= {current_house_weight_gradient}"
        )

    def log_current_house_weights_before_update(
        self,
        current_house_weights,
        learning_rate,
        current_house_weight_gradient,
    ):
        if not self.enabled:
            return
        print("\n/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */")
        print(
            "\nCURRENT_HOUSE_WEIGHTS"
            "\nCALCULE :\ncurrent_house_weights - learning_rate * current_house_weight_gradient"
            f"\n{current_house_weights} - {learning_rate} * {current_house_weight_gradient}"
            f"\n{current_house_weights} - {learning_rate * current_house_weight_gradient}"
            "\n--------------------------------------------------------------"
        )

    def log_current_house_weights_after_update(self, current_house_weights):
        if not self.enabled:
            return
        print(f"{current_house_weights}")

    def log_house_disciplines_weights(self, house_disciplines_weights):
        if not self.enabled:
            return
        print(f"\nHOUSE_DISCIPLINES_WEIGHTS : \n{house_disciplines_weights}")
