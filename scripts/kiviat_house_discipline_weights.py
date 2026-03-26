#!/usr/bin/env python3
import argparse
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.interpolate import CubicSpline
except Exception:
    CubicSpline = None


DEFAULT_FIG_WIDTH = 16.0
DEFAULT_FIG_HEIGHT = 9.0
DEFAULT_DPI = 120

HOUSE_COLOR_BY_NAME = {
    "Gryffindor": "#D62839",
    "Hufflepuff": "#F6AA1C",
    "Ravenclaw": "#2E5CB8",
    "Slytherin": "#2A9D5B",
}

KNOWN_SHORT_LABEL_BY_DISCIPLINE = {
    "Defense Against the Dark Arts": "Defense\nDark Arts",
    "Care of Magical Creatures": "Magical\nCreatures",
    "History of Magic": "History\nof Magic",
    "Muggle Studies": "Muggle\nStudies",
    "Ancient Runes": "Ancient\nRunes",
}


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Generer un diagramme de Kiviat des poids des disciplines "
            "par maison a partir de weights.json."
        )
    )
    argument_parser.add_argument(
        "trained_parameter_json_file_path",
        nargs="?",
        default="weights.json",
        help="Chemin du fichier de poids (defaut: weights.json).",
    )
    argument_parser.add_argument(
        "--out",
        "-o",
        dest="output_image_path",
        default="visuals/kiviat_house_discipline_weights.png",
        help="Chemin du PNG de sortie (defaut: visuals/kiviat_house_discipline_weights.png).",
    )
    argument_parser.add_argument(
        "--width",
        type=float,
        default=DEFAULT_FIG_WIDTH,
        help="Largeur de la figure en pouces (defaut: 16).",
    )
    argument_parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_FIG_HEIGHT,
        help="Hauteur de la figure en pouces (defaut: 9).",
    )
    argument_parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Resolution de sortie PNG (defaut: 120).",
    )
    argument_parser.add_argument(
        "--include-bias",
        action="store_true",
        help="Inclure l'axe bias dans le Kiviat (desactive par defaut).",
    )
    argument_parser.add_argument(
        "--show",
        action="store_true",
        help="Afficher la fenetre matplotlib en plus de la sauvegarde.",
    )
    argument_parser.add_argument(
        "--smooth-points-per-segment",
        type=int,
        default=24,
        help=(
            "Nombre de points de lissage entre deux disciplines "
            "(defaut: 24). Mettre 1 pour des segments droits."
        ),
    )
    return argument_parser.parse_args()


def load_house_discipline_weights(trained_parameter_json_file_path, include_bias):
    with open(trained_parameter_json_file_path, "r") as parameter_file:
        trained_parameter_bundle = json.load(parameter_file)

    missing_key_names = [
        key_name
        for key_name in ("thetas", "features", "inv_house_map")
        if key_name not in trained_parameter_bundle
    ]
    if missing_key_names:
        raise KeyError(f"Cles manquantes dans le JSON: {missing_key_names}")

    house_discipline_weights_with_bias = np.array(trained_parameter_bundle["thetas"], dtype=float)
    discipline_names = list(trained_parameter_bundle["features"])
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle["inv_house_map"].items()
    }

    if house_discipline_weights_with_bias.ndim != 2:
        raise ValueError("Le champ 'thetas' doit etre une matrice 2D.")

    expected_discipline_plus_bias_count = len(discipline_names) + 1
    actual_discipline_plus_bias_count = house_discipline_weights_with_bias.shape[1]
    if actual_discipline_plus_bias_count != expected_discipline_plus_bias_count:
        raise ValueError(
            "Incoherence entre 'thetas' et 'features': "
            f"{actual_discipline_plus_bias_count} colonnes pour thetas, "
            f"{len(discipline_names)} disciplines."
        )

    sorted_house_codes = sorted(house_name_by_code.keys())
    if not sorted_house_codes:
        raise ValueError("Aucune maison trouvee dans inv_house_map.")

    highest_house_code = max(sorted_house_codes)
    house_count_in_weights = house_discipline_weights_with_bias.shape[0]
    if highest_house_code >= house_count_in_weights:
        raise ValueError(
            "Incoherence entre les lignes de 'thetas' et les codes maison dans inv_house_map."
        )

    house_names_in_code_order = [house_name_by_code[house_code] for house_code in sorted_house_codes]
    house_discipline_weights_in_code_order = house_discipline_weights_with_bias[sorted_house_codes, :]

    if include_bias:
        axis_names = ["bias"] + discipline_names
        house_discipline_weights = house_discipline_weights_in_code_order
    else:
        axis_names = discipline_names
        house_discipline_weights = house_discipline_weights_in_code_order[:, 1:]

    return house_names_in_code_order, axis_names, house_discipline_weights


def make_short_axis_label(axis_name):
    if axis_name in KNOWN_SHORT_LABEL_BY_DISCIPLINE:
        return KNOWN_SHORT_LABEL_BY_DISCIPLINE[axis_name]

    if len(axis_name) <= 14:
        return axis_name

    wrapped_lines = textwrap.wrap(axis_name, width=14, break_long_words=False)
    if len(wrapped_lines) == 1:
        return wrapped_lines[0]
    return "\n".join(wrapped_lines[:2])


def close_polar_curve(axis_angles, radial_values):
    closed_axis_angles = np.concatenate([axis_angles, [2.0 * np.pi]])
    closed_radial_values = np.concatenate([radial_values, [radial_values[0]]])
    return closed_axis_angles, closed_radial_values


def smooth_polar_curve(axis_angles, radial_values, smooth_points_per_segment):
    closed_axis_angles, closed_radial_values = close_polar_curve(axis_angles, radial_values)

    safe_smooth_points_per_segment = max(1, int(smooth_points_per_segment))
    if safe_smooth_points_per_segment == 1:
        return closed_axis_angles, closed_radial_values

    dense_point_count = max(len(axis_angles) * safe_smooth_points_per_segment, len(axis_angles) + 1)
    dense_axis_angles = np.linspace(0.0, 2.0 * np.pi, dense_point_count + 1)

    if CubicSpline is not None:
        smoothing_function = CubicSpline(
            closed_axis_angles,
            closed_radial_values,
            bc_type="periodic",
        )
        dense_radial_values = smoothing_function(dense_axis_angles)
    else:
        dense_radial_values = np.interp(
            dense_axis_angles,
            closed_axis_angles,
            closed_radial_values,
        )

    dense_radial_values = np.clip(dense_radial_values, 0.0, 2.0)
    return dense_axis_angles, dense_radial_values


def plot_kiviat(
    house_names,
    axis_names,
    house_discipline_weights,
    output_image_path,
    width,
    height,
    dpi,
    smooth_points_per_segment,
):
    axis_count = len(axis_names)
    if axis_count < 3:
        raise ValueError("Il faut au moins 3 axes pour dessiner un diagramme de Kiviat.")

    max_absolute_weight = float(np.max(np.abs(house_discipline_weights)))
    if max_absolute_weight < 1e-12:
        max_absolute_weight = 1.0

    normalized_house_discipline_weights = house_discipline_weights / max_absolute_weight
    radial_house_discipline_weights = normalized_house_discipline_weights + 1.0

    axis_angles = np.linspace(0, 2 * np.pi, axis_count, endpoint=False)
    short_axis_names = [make_short_axis_label(axis_name) for axis_name in axis_names]
    closed_axis_angles_for_baseline, _ = close_polar_curve(axis_angles, np.zeros(axis_count))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(width, height), facecolor="#BCC3CE")
    axis = figure.add_subplot(111, projection="polar", facecolor="#D8DEE7")

    figure.subplots_adjust(top=0.78, bottom=0.18, left=0.10, right=0.90)
    figure.suptitle(
        "Diagramme de Kiviat - Poids des disciplines par maison",
        fontsize=22,
        fontweight="bold",
        color="#111827",
        y=0.978,
    )
    figure.text(
        0.5,
        0.922,
        "Rayon 1.0 = poids nul | >1.0 = poids positif | <1.0 = poids negatif",
        ha="center",
        va="center",
        fontsize=11,
        color="#1F2937",
    )
    figure.text(
        0.5,
        0.886,
        f"Normalisation globale: max |poids| = {max_absolute_weight:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    axis.set_theta_offset(np.pi / 2.0)
    axis.set_theta_direction(-1)

    axis.set_xticks(axis_angles)
    axis.set_xticklabels(short_axis_names, fontsize=11, color="#1F2937")
    axis.tick_params(axis="x", pad=22)

    axis.set_ylim(0.0, 2.0)
    axis.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axis.set_yticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], fontsize=10, color="#334155")
    axis.tick_params(axis="y", pad=8)

    axis.yaxis.grid(True, color="#A9B4C1", linewidth=1.0, alpha=0.9)
    axis.xaxis.grid(True, color="#B7C2CF", linestyle="--", linewidth=0.9, alpha=0.9)
    axis.spines["polar"].set_color("#4F5F73")
    axis.spines["polar"].set_linewidth(1.2)

    axis.plot(
        closed_axis_angles_for_baseline,
        np.full_like(closed_axis_angles_for_baseline, 1.0),
        color="#2D3748",
        linestyle=(0, (4, 2)),
        linewidth=1.3,
        alpha=0.85,
    )

    fallback_colors = plt.cm.Set2(np.linspace(0, 1, len(house_names)))
    for house_index, house_name in enumerate(house_names):
        house_color = HOUSE_COLOR_BY_NAME.get(house_name, fallback_colors[house_index])
        raw_radial_values = radial_house_discipline_weights[house_index, :]
        smooth_axis_angles, smooth_radial_values = smooth_polar_curve(
            axis_angles,
            raw_radial_values,
            smooth_points_per_segment,
        )

        axis.plot(
            smooth_axis_angles,
            smooth_radial_values,
            color=house_color,
            linewidth=2.5,
            label=house_name,
            alpha=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.fill(smooth_axis_angles, smooth_radial_values, color=house_color, alpha=0.15)
        axis.scatter(axis_angles, raw_radial_values, color=house_color, s=16, alpha=0.95, zorder=3)

    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(4, len(house_names)),
        frameon=True,
        fontsize=11,
    )
    legend.get_frame().set_facecolor("#EEF2F7")
    legend.get_frame().set_edgecolor("#A8B4C3")

    absolute_output_image_path = os.path.abspath(output_image_path)
    output_directory_path = os.path.dirname(absolute_output_image_path)
    if output_directory_path:
        os.makedirs(output_directory_path, exist_ok=True)

    figure.savefig(absolute_output_image_path, dpi=dpi)
    plt.close(figure)
    return absolute_output_image_path


def main():
    try:
        cli_arguments = parse_command_line_arguments()

        (
            house_names,
            axis_names,
            house_discipline_weights,
        ) = load_house_discipline_weights(
            cli_arguments.trained_parameter_json_file_path,
            cli_arguments.include_bias,
        )

        absolute_output_image_path = plot_kiviat(
            house_names,
            axis_names,
            house_discipline_weights,
            cli_arguments.output_image_path,
            cli_arguments.width,
            cli_arguments.height,
            cli_arguments.dpi,
            cli_arguments.smooth_points_per_segment,
        )
        print(f"→ Diagramme de Kiviat enregistre dans {absolute_output_image_path}")

        if cli_arguments.show:
            output_image = plt.imread(absolute_output_image_path)
            preview_figure, preview_axis = plt.subplots(figsize=(10, 6))
            preview_axis.imshow(output_image)
            preview_axis.axis("off")
            preview_figure.canvas.manager.set_window_title("Kiviat - poids disciplines")
            plt.show()
            plt.close(preview_figure)

    except Exception as exception:
        print(f"Une erreur est survenue : {exception}")


if __name__ == "__main__":
    main()
