import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

import logreg_train as train_module


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Anime l'evolution des poids one-vs-rest pour comprendre "
            "comment les poids sont appris par maison et discipline."
        )
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
        help="Taux d'apprentissage (defaut: 0.01)."
    )
    argument_parser.add_argument(
        "--iterations", "-n",
        dest="iteration_count",
        type=int,
        default=1000,
        help="Nombre d'iterations par maison (defaut: 1000)."
    )
    argument_parser.add_argument(
        "--frame-step",
        dest="frame_step",
        type=int,
        default=10,
        help="Conserve une frame toutes les N updates (defaut: 10)."
    )
    argument_parser.add_argument(
        "--interval-ms",
        dest="frame_interval_ms",
        type=int,
        default=120,
        help="Intervalle entre deux frames en millisecondes (defaut: 120)."
    )
    argument_parser.add_argument(
        "--max-preview-frames",
        dest="max_preview_frame_count",
        type=int,
        default=450,
        help="Nombre maximum de frames en affichage ecran (defaut: 450)."
    )
    argument_parser.add_argument(
        "--figure-scale",
        dest="figure_scale",
        type=float,
        default=1.0,
        help=(
            "Facteur d'echelle pour la figure 16:9 "
            "(1.0 -> 16x9 pouces, 1.5 -> 24x13.5)."
        ),
    )
    argument_parser.add_argument(
        "--gif-final-frame-hold-ms",
        dest="gif_final_frame_hold_ms",
        type=int,
        default=2000,
        help=(
            "Duree d'affichage de la derniere frame (GIF) en ms "
            "pour mieux marquer la fin (defaut: 2000)."
        ),
    )
    argument_parser.add_argument(
        "--save",
        dest="output_animation_path",
        default=None,
        help="Chemin de sortie optionnel (ex: visuals/weights_evolution.gif)."
    )
    argument_parser.add_argument(
        "--no-show",
        dest="no_show",
        action="store_true",
        help="Ne pas ouvrir la fenetre matplotlib."
    )
    return argument_parser.parse_args()


def collect_weight_history(
    students_discipline_scores_with_bias,
    assigned_house_codes_by_student,
    learning_rate,
    iteration_count,
    frame_step,
):
    student_count, discipline_plus_bias_count = students_discipline_scores_with_bias.shape
    unique_house_codes = np.unique(assigned_house_codes_by_student)
    house_count = len(unique_house_codes)

    house_discipline_weights = np.zeros((house_count, discipline_plus_bias_count))
    history = [(house_discipline_weights.copy(), None, 0, 0)]

    global_step = 0
    step_stride = max(1, int(frame_step))

    for current_house_code in unique_house_codes:
        current_house_weights = np.zeros(discipline_plus_bias_count)
        is_student_assigned_to_current_house = (
            assigned_house_codes_by_student == current_house_code
        ).astype(float)

        for iteration_in_house in range(1, iteration_count + 1):
            predicted_probability_of_current_house = train_module.compute_sigmoid(
                students_discipline_scores_with_bias.dot(current_house_weights)
            )
            current_house_weight_gradient = (
                1 / student_count
            ) * students_discipline_scores_with_bias.T.dot(
                predicted_probability_of_current_house - is_student_assigned_to_current_house
            )
            current_house_weights -= learning_rate * current_house_weight_gradient
            house_discipline_weights[int(current_house_code), :] = current_house_weights

            global_step += 1
            keep_frame = (
                iteration_in_house == 1
                or iteration_in_house == iteration_count
                or global_step % step_stride == 0
            )
            if keep_frame:
                history.append(
                    (
                        house_discipline_weights.copy(),
                        int(current_house_code),
                        iteration_in_house,
                        global_step,
                    )
                )

    return history


def compute_axis_limits(series_by_house):
    minimum_value = float(np.min(series_by_house))
    maximum_value = float(np.max(series_by_house))

    if abs(maximum_value - minimum_value) < 1e-9:
        minimum_value -= 1e-3
        maximum_value += 1e-3

    padding = 0.1 * (maximum_value - minimum_value)
    return minimum_value - padding, maximum_value + padding


def downsample_history_for_preview(history, max_preview_frame_count):
    if len(history) <= max_preview_frame_count:
        return history

    if max_preview_frame_count < 2:
        return [history[0], history[-1]]

    stride = int(np.ceil((len(history) - 1) / (max_preview_frame_count - 1)))
    reduced_history = history[::stride]
    if reduced_history[-1][3] != history[-1][3]:
        reduced_history.append(history[-1])
    return reduced_history


def enforce_finite_gif_playback(gif_file_path, final_frame_hold_ms):
    with Image.open(gif_file_path) as gif_image:
        gif_frame_count = getattr(gif_image, "n_frames", 1)
        frame_list = []
        frame_duration_list = []

        for frame_index in range(gif_frame_count):
            gif_image.seek(frame_index)
            frame_list.append(gif_image.copy())
            frame_duration_list.append(int(gif_image.info.get("duration", 100)))

    if not frame_list:
        return

    safe_final_frame_hold_ms = max(100, int(final_frame_hold_ms))
    frame_duration_list[-1] = max(frame_duration_list[-1], safe_final_frame_hold_ms)

    frame_list[0].save(
        gif_file_path,
        save_all=True,
        append_images=frame_list[1:],
        duration=frame_duration_list,
        loop=1,
        optimize=False,
    )


def build_animation(
    history,
    house_name_by_code,
    discipline_names,
    iteration_count,
    frame_interval_ms,
    figure_scale,
):
    global_steps = np.array([snapshot[3] for snapshot in history], dtype=int)
    weight_history = np.array([snapshot[0] for snapshot in history], dtype=float)

    bias_history_by_house = weight_history[:, :, 0]
    discipline_weight_history_by_house = weight_history[:, :, 1:]

    house_count = weight_history.shape[1]
    discipline_count = discipline_weight_history_by_house.shape[2]

    house_codes = list(range(house_count))
    house_labels = [house_name_by_code[int(house_code)] for house_code in house_codes]

    x_max = int(global_steps[-1]) if len(global_steps) > 1 else 1
    bias_y_min, bias_y_max = compute_axis_limits(bias_history_by_house)

    discipline_axis_limits = [
        compute_axis_limits(discipline_weight_history_by_house[:, :, discipline_index])
        for discipline_index in range(discipline_count)
    ]

    column_count = 4
    discipline_row_count = int(np.ceil(discipline_count / column_count))
    total_row_count = 1 + discipline_row_count

    safe_figure_scale = max(0.25, float(figure_scale))
    figure_width_inch = 16.0 * safe_figure_scale
    figure_height_inch = 9.0 * safe_figure_scale
    figure = plt.figure(figsize=(figure_width_inch, figure_height_inch))
    grid = figure.add_gridspec(
        total_row_count,
        column_count,
        hspace=0.42,
        wspace=0.22,
        height_ratios=[2.8] + [1.8] * discipline_row_count,
    )
    figure.subplots_adjust(top=0.86, bottom=0.06, left=0.05, right=0.985)

    figure.suptitle("Evolution des poids pendant l'entrainement one-vs-rest", y=0.985)
    status_axis = figure.add_axes([0.05, 0.935, 0.935, 0.03], frameon=False)
    status_axis.set_axis_off()
    status_text = status_axis.text(
        0.5,
        0.5,
        "",
        transform=status_axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )

    bias_axis = figure.add_subplot(grid[0, :])
    bias_axis.set_title("Bias par maison")
    bias_axis.set_ylabel("Valeur du bias")
    bias_axis.set_xlim(0, max(1, x_max))
    bias_axis.set_ylim(bias_y_min, bias_y_max)
    bias_axis.grid(alpha=0.25)

    bias_lines = []
    bias_markers = []
    house_colors = []
    for house_label in house_labels:
        (bias_line,) = bias_axis.plot([], [], linewidth=2.0, label=house_label)
        (bias_marker,) = bias_axis.plot([], [], marker="o", markersize=4, linestyle="None")
        bias_marker.set_color(bias_line.get_color())

        bias_lines.append(bias_line)
        bias_markers.append(bias_marker)
        house_colors.append(bias_line.get_color())

    bias_axis.legend(loc="upper right", ncols=2, fontsize=9)

    discipline_lines = []

    def compute_row_column_spans(plot_count_in_row):
        if plot_count_in_row == column_count:
            return [(column_index, column_index + 1) for column_index in range(column_count)]
        if column_count == 4 and plot_count_in_row == 1:
            return [(1, 2)]
        if column_count == 4 and plot_count_in_row == 2:
            return [(0, 2), (2, 4)]

        starting_column = (column_count - plot_count_in_row) // 2
        return [
            (starting_column + offset, starting_column + offset + 1)
            for offset in range(plot_count_in_row)
        ]

    next_discipline_index = 0
    for row_index in range(discipline_row_count):
        plot_count_in_row = min(column_count, discipline_count - next_discipline_index)

        if plot_count_in_row == 1:
            discipline_name = discipline_names[next_discipline_index]
            single_cell_box = grid[1 + row_index, 0].get_position(figure)
            full_row_box = grid[1 + row_index, :].get_position(figure)
            centered_left = full_row_box.x0 + (full_row_box.width - single_cell_box.width) / 2
            axis = figure.add_axes(
                [centered_left, full_row_box.y0, single_cell_box.width, single_cell_box.height]
            )

            y_min, y_max = discipline_axis_limits[next_discipline_index]
            axis.set_xlim(0, max(1, x_max))
            axis.set_ylim(y_min, y_max)
            axis.grid(alpha=0.2)
            axis.set_title(discipline_name, fontsize=9)
            axis.tick_params(axis="x", labelsize=8)
            axis.tick_params(axis="y", labelsize=8)

            lines_for_discipline = []
            for house_index in range(house_count):
                (line_object,) = axis.plot([], [], linewidth=1.3, color=house_colors[house_index])
                lines_for_discipline.append(line_object)

            discipline_lines.append(lines_for_discipline)
            next_discipline_index += 1
            continue

        row_column_spans = compute_row_column_spans(plot_count_in_row)
        occupied_columns = set()

        for start_column, end_column in row_column_spans:
            occupied_columns.update(range(start_column, end_column))

            discipline_name = discipline_names[next_discipline_index]
            axis = figure.add_subplot(grid[1 + row_index, start_column:end_column])

            y_min, y_max = discipline_axis_limits[next_discipline_index]
            axis.set_xlim(0, max(1, x_max))
            axis.set_ylim(y_min, y_max)
            axis.grid(alpha=0.2)
            axis.set_title(discipline_name, fontsize=9)
            axis.tick_params(axis="x", labelsize=8)
            axis.tick_params(axis="y", labelsize=8)

            lines_for_discipline = []
            for house_index in range(house_count):
                (line_object,) = axis.plot([], [], linewidth=1.3, color=house_colors[house_index])
                lines_for_discipline.append(line_object)

            discipline_lines.append(lines_for_discipline)
            next_discipline_index += 1

        for column_index in range(column_count):
            if column_index in occupied_columns:
                continue
            empty_axis = figure.add_subplot(grid[1 + row_index, column_index])
            empty_axis.axis("off")

    def update(frame_index):
        x_values = global_steps[: frame_index + 1]

        for house_index in range(house_count):
            bias_values = bias_history_by_house[: frame_index + 1, house_index]
            bias_lines[house_index].set_data(x_values, bias_values)
            bias_markers[house_index].set_data([x_values[-1]], [bias_values[-1]])

        for discipline_index in range(discipline_count):
            for house_index in range(house_count):
                discipline_values = discipline_weight_history_by_house[
                    : frame_index + 1, house_index, discipline_index
                ]
                discipline_lines[discipline_index][house_index].set_data(x_values, discipline_values)

        _weights_snapshot, current_house_code, iteration_in_house, global_step = history[frame_index]
        if current_house_code is None:
            status_text.set_text("Etape initiale: tous les poids sont a 0")
        else:
            current_house_name = house_name_by_code[int(current_house_code)]
            status_text.set_text(
                f"Etape {global_step} | Maison en cours: {current_house_name} "
                f"| Iteration maison: {iteration_in_house}/{iteration_count}"
            )

        artist_list = [status_text, *bias_lines, *bias_markers]
        for lines_for_discipline in discipline_lines:
            artist_list.extend(lines_for_discipline)
        return artist_list

    animation = FuncAnimation(
        figure,
        update,
        frames=len(history),
        interval=frame_interval_ms,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )
    return figure, animation


def main():
    try:
        cli_arguments = parse_command_line_arguments()

        (
            student_discipline_scores,
            assigned_house_codes_by_student,
            _house_code_by_name,
            house_name_by_code,
            discipline_names,
        ) = train_module.load_and_prepare_dataset(cli_arguments.input_csv_path)

        (
            standardized_discipline_scores,
            _average_score_by_discipline,
            _standard_deviation_by_discipline,
        ) = train_module.standardize_discipline_scores(student_discipline_scores)

        student_count = standardized_discipline_scores.shape[0]
        students_discipline_scores_with_bias = np.hstack(
            [np.ones((student_count, 1)), standardized_discipline_scores]
        )

        history = collect_weight_history(
            students_discipline_scores_with_bias,
            assigned_house_codes_by_student,
            cli_arguments.learning_rate,
            cli_arguments.iteration_count,
            cli_arguments.frame_step,
        )

        if cli_arguments.no_show and not cli_arguments.output_animation_path:
            print("Mode --no-show sans --save: rien a afficher, rien a enregistrer.")
            return

        history_for_animation = history
        if not cli_arguments.no_show:
            history_for_animation = downsample_history_for_preview(
                history,
                cli_arguments.max_preview_frame_count,
            )
            if len(history_for_animation) < len(history):
                print(
                    f"Apercu allege: {len(history)} -> {len(history_for_animation)} frames "
                    f"(max={cli_arguments.max_preview_frame_count})"
                )

        figure, animation = build_animation(
            history_for_animation,
            house_name_by_code,
            discipline_names,
            cli_arguments.iteration_count,
            cli_arguments.frame_interval_ms,
            cli_arguments.figure_scale,
        )

        if cli_arguments.output_animation_path:
            try:
                absolute_output_animation_path = os.path.abspath(
                    cli_arguments.output_animation_path
                )
                output_directory_path = os.path.dirname(absolute_output_animation_path)
                if output_directory_path:
                    os.makedirs(output_directory_path, exist_ok=True)
                animation.save(absolute_output_animation_path)

                if absolute_output_animation_path.lower().endswith(".gif"):
                    enforce_finite_gif_playback(
                        absolute_output_animation_path,
                        cli_arguments.gif_final_frame_hold_ms,
                    )

                if os.path.isfile(absolute_output_animation_path):
                    output_file_size = os.path.getsize(absolute_output_animation_path)
                    print(
                        f"Animation enregistree dans {absolute_output_animation_path} "
                        f"({output_file_size} octets)"
                    )
                else:
                    print(
                        "Animation terminee mais fichier introuvable apres sauvegarde: "
                        f"{absolute_output_animation_path}"
                    )
            except Exception as exception:
                print(f"Impossible d'enregistrer l'animation: {exception}")
        else:
            print("Aucun fichier enregistre: ajoute --save <chemin>.gif")

        if not cli_arguments.no_show:
            plt.show(block=True)
        else:
            plt.close(figure)

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
