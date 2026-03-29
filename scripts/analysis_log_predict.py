import numpy as np


class AnalysisPredictLogger:
    """Logger verbeux pour analyser la phase de prediction."""

    GRAPHICAL_SEPARATOR = "/*   -'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-'-,-',-'   */"
    VALUE_SEPARATOR = "--------------------------------------------------------------"

    def __init__(self, enabled=False):
        self.enabled = enabled

    def _log_graphical_separator(self):
        print(f"\n{self.GRAPHICAL_SEPARATOR}")

    def log_students_discipline_scores(self, students_discipline_scores):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print("\nSTUDENTS_DISCIPLINE_SCORES")
        print(f"= \n{students_discipline_scores}")

    def log_standardized_students_discipline_scores(
        self,
        students_discipline_scores,
        average_discipline_scores,
        discipline_standard_deviations,
        standardized_students_discipline_scores,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nSTANDARDIZED_STUDENTS_DISCIPLINE_SCORES"
            "\nCALCULE :"
            "\n(students_discipline_scores - average_discipline_scores) / discipline_standard_deviations"
            f"\n({students_discipline_scores.to_numpy(dtype=float)} - {np.array(average_discipline_scores, dtype=float)}) / {np.array(discipline_standard_deviations, dtype=float)}"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {standardized_students_discipline_scores}"
        )

    def log_students_discipline_scores_with_bias(
        self,
        students_count,
        standardized_students_discipline_scores,
        students_discipline_scores_with_bias,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nSTUDENTS_DISCIPLINE_SCORES_WITH_BIAS"
            "\nCALCULE :"
            "\nnp.hstack([np.ones((students_count, 1)), standardized_students_discipline_scores])"
            f"\nnp.hstack([np.ones(({students_count}, 1)), {standardized_students_discipline_scores}])"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {students_discipline_scores_with_bias}"
        )

    def log_house_scores_before_value_limit_for_all_students(
        self,
        students_discipline_scores_with_bias,
        house_discipline_weights_with_bias,
        house_scores_before_value_limit_for_all_students,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nHOUSE_SCORES_BEFORE_VALUE_LIMIT_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\nstudents_discipline_scores_with_bias.dot(house_discipline_weights_with_bias.T)"
            f"\n{students_discipline_scores_with_bias}.dot({house_discipline_weights_with_bias}.T)"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {house_scores_before_value_limit_for_all_students}"
        )

    def log_house_scores_after_value_limit_for_all_students(
        self,
        house_scores_before_value_limit_for_all_students,
        house_scores_after_value_limit_for_all_students,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nHOUSE_SCORES_AFTER_VALUE_LIMIT_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\nnp.clip(house_scores_before_value_limit_for_all_students, -500, 500)"
            f"\nnp.clip({house_scores_before_value_limit_for_all_students}, -500, 500)"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {house_scores_after_value_limit_for_all_students}"
        )

    def log_house_probability_scores_for_all_students(
        self,
        house_scores_after_value_limit_for_all_students,
        house_probability_scores_for_all_students,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nHOUSE_PROBABILITY_SCORES_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\n1 / (1 + np.exp(-house_scores_after_value_limit_for_all_students))"
            f"\n1 / (1 + np.exp(-{house_scores_after_value_limit_for_all_students}))"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {house_probability_scores_for_all_students}"
        )

    def log_predicted_house_codes_for_all_students(
        self,
        house_probability_scores_for_all_students,
        predicted_house_codes_for_all_students,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nPREDICTED_HOUSE_CODES_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\nnp.argmax(house_probability_scores_for_all_students, axis=1)"
            f"\nnp.argmax({house_probability_scores_for_all_students}, axis=1)"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {predicted_house_codes_for_all_students}"
        )

    def log_predicted_house_names_for_all_students(
        self,
        house_name_by_code,
        predicted_house_codes_for_all_students,
        predicted_house_names_for_all_students,
    ):
        if not self.enabled:
            return
        self._log_graphical_separator()
        print(
            "\nPREDICTED_HOUSE_NAMES_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\n[house_name_by_code[int(house_code)] for house_code in predicted_house_codes_for_all_students]"
            f"\nhouse_name_by_code = {house_name_by_code}"
            f"\n[house_name_by_code[int(house_code)] for house_code in {predicted_house_codes_for_all_students}]"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {predicted_house_names_for_all_students}"
        )
        self._log_graphical_separator()
        print(
            "\nNUMBER_OF_PREDICTED_HOUSE_NAMES_FOR_ALL_STUDENTS"
            "\nCALCULE :"
            "\nlen(predicted_house_names_for_all_students)"
            f"\nlen({predicted_house_names_for_all_students})"
            f"\n{self.VALUE_SEPARATOR}"
            f"\n= {len(predicted_house_names_for_all_students)}"
        )
