import argparse
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import logging
import os
import json
import numpy as np
import fillers as fl
import sys
import plots as pl
import config
import pdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def setup_argument_parser():
    """Set up the command line argument parser."""
    parser = argparse.ArgumentParser(description="Generate a quantum benchmark report.")
    parser.add_argument(
        "--experiment-left",
        type=str,
        default="rb-1306",
        help="Directory containing the experiment data.",
    )
    parser.add_argument(
        "--experiment-right",
        type=str,
        default="BASELINE",
        help="Directory containing the experiment data.",
    )
    parser.add_argument(
        "--data-right",
        type=str,
        default="numpy",
        help="Directory containing the experiment data.",
    )
    parser.add_argument(
        "--data-left",
        type=str,
        default="nqch-sim",
        help="Directory containing the experiment data.",
    )

    # Plot toggles (default: True). Use --no-<flag> to disable.
    parser.add_argument("--t1-plot", dest="t1_plot", action="store_true", default=True)
    parser.add_argument("--no-t1-plot", dest="t1_plot", action="store_false")
    parser.add_argument(
        "--mermin-5-plot", dest="mermin_5_plot", action="store_true", default=True
    )
    parser.add_argument(
        "--no-mermin-5-plot", dest="mermin_5_plot", action="store_false"
    )
    parser.add_argument(
        "--reuploading-plot", dest="reuploading_plot", action="store_true", default=True
    )
    parser.add_argument(
        "--no-reuploading-plot", dest="reuploading_plot", action="store_false"
    )
    parser.add_argument(
        "--grover-plot", dest="grover_plot", action="store_true", default=True
    )
    parser.add_argument("--no-grover-plot", dest="grover_plot", action="store_false")
    parser.add_argument(
        "--ghz-plot", dest="ghz_plot", action="store_true", default=True
    )
    parser.add_argument("--no-ghz-plot", dest="ghz_plot", action="store_false")

    return parser


def add_stat_changes(current, baseline):
    """
    Returns a dict with average, min, max, median and their changes vs baseline.
    """

    def get_change(curr, base):
        if curr is None or base is None:
            return None
        try:
            diff = float(curr) - float(base)
            if base == 0:
                return None
            percent = (diff / float(base)) * 100
            if percent > 0:
                return f"(+{percent:.2f}\\%)"
            elif percent < 0:
                return f"(-{percent:.2f}\\%)"
            else:
                return "-"
            # return f"{diff:+.4g} ({percent:+.2f}\\%)"
        except Exception as e:
            print(f"Error calculating change: {e}")
            sys.exit(1)
            return None

    result = {}
    for key in ["average", "min", "max", "median"]:
        curr_val = current.get(key)
        base_val = baseline.get(key)
        result[key] = curr_val
        result[f"{key}_change"] = get_change(curr_val, base_val)
    # print(f"Stat fidelity changes: {result}")
    return result


def prepare_template_context(args):
    """
    Prepare a complete template context for the benchmarking report.
    """
    logging.info("Preparing context for full benchmarking report.")

    # Load experiment metadata from mega.json
    meta_json_path = Path("data") / args.experiment_left / "meta.json"
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
    logging.info("Loaded experiment metadata from %s", meta_json_path)

    ######### Fidelity statistics and changes
    stat_fidelity = fl.get_stat_fidelity(args.experiment_left)
    stat_fidelity_baseline = fl.get_stat_fidelity(args.experiment_right)
    stat_fidelity_with_improvement = add_stat_changes(
        stat_fidelity, stat_fidelity_baseline
    )
    logging.info("Prepared stat_fidelity and stat_fidelity_with_improvement")

    ##########Pulse Fidelity statistics and changes
    stat_pulse_fidelity = fl.get_stat_pulse_fidelity(args.experiment_left)
    stat_pulse_fidelity_baseline = fl.get_stat_pulse_fidelity(args.experiment_right)
    stat_pulse_fidelity_with_improvement = add_stat_changes(
        stat_pulse_fidelity, stat_pulse_fidelity_baseline
    )
    logging.info(
        "Prepared stat_pulse_fidelity and stat_pulse_fidelity_with_improvement"
    )

    ######### T1 statistics and changes
    stat_t1 = fl.get_stat_t12(args.experiment_left, "t1")
    stat_t1_baseline = fl.get_stat_t12(args.experiment_right, "t1")
    stat_t1_with_improvement = add_stat_changes(stat_t1, stat_t1_baseline)
    logging.info("Prepared stat_t1 and stat_t1_with_improvement")

    ######### T2 statistics and changes
    stat_t2 = fl.get_stat_t12(args.experiment_left, "t2")
    stat_t2_baseline = fl.get_stat_t12(args.experiment_right, "t2")
    stat_t2_with_improvement = add_stat_changes(stat_t2, stat_t2_baseline)
    logging.info("Prepared stat_t2 and stat_t2_with_improvement")

    context = {
        "experiment_name": meta_data.get("title", "Unknown Title"),
        "platform": meta_data.get("platform", "Unknown Platform"),
        #
        "start_time": meta_data.get("start-time", "Unknown Start Time"),
        "end_time": meta_data.get("start-time", "Unknown Start Time"),
        #
        "report_of_changes": "\\textcolor{green}{More report of changes (from software).}",
        #
        "stat_fidelity": stat_fidelity_with_improvement,
        "stat_fidelity_baseline": stat_fidelity_baseline,
        #
        "stat_pulse_fidelity": stat_pulse_fidelity_with_improvement,
        "stat_pulse_fidelity_baseline": stat_pulse_fidelity_baseline,
        #
        "stat_t1": stat_t1_with_improvement,
        "stat_t1_baseline": stat_t1_baseline,
        #
        "stat_t2": stat_t2_with_improvement,
        "stat_t2_baseline": stat_t2_baseline,
        #
        "new_version": fl.context_new_version(args, meta_data),
        "control_version": fl.context_control_version(args),
        #
        "new_fidelity": fl.context_fidelity(args.experiment_left),
        "control_fidelity": fl.context_fidelity(args.experiment_right),
        #
        "plot_exp": pl.plot_fidelity_graph(
            args.experiment_left, config.connectivity, config.pos
        ),
        "plot_baseline": pl.plot_fidelity_graph(
            args.experiment_right, config.connectivity, config.pos
        ),
        #
        "plot_chevron_swap_0": pl.plot_chevron_swap_coupler(
            qubit_number=0,
            data_dir="data/DEMODATA/",
            output_path="build/",
        ),
        #
        "plot_swap_coupler": pl.plot_swap_coupler(
            qubit_number=0,
            data_dir="data/DEMODATA/",
            output_path="build/",
        ),
    }
    logging.info("Basic context dictionary prepared")

    # # Add additional plots if needed
    # context["grid_coupler_is_set"] = False
    # grid_coupler_plots = pl.prepare_grid_coupler(
    #     max_number=2,
    #     data_dir="data/DEMODATA",
    #     baseline_dir="data/DEMODATA",
    #     output_path="build/",
    # )
    # context["plot_grid_coupler"] = grid_coupler_plots
    # logging.info("Added grid_coupler plots to context")

    # # Add additional chevron_swap_coupler plots if needed
    # context["chevron_swap_coupler_is_set"] = False
    # chevron_swap_coupler_plots = pl.prepare_grid_chevron_swap_coupler(
    #     max_number=2,
    #     data_dir="data/DEMODATA",
    #     baseline_dir="data/DEMODATA",
    #     output_path="build/",
    # )
    # context["plot_chevron_swap_coupler"] = chevron_swap_coupler_plots
    # logging.info("Added chevron_swap_coupler plots to context")

    # ######### T1 PLOT
    if args.t1_plot == True:
        context["t1_plot_is_set"] = True
        t1_plot = pl.prepare_grid_t1_plts(
            max_number=2,
            data_dir="data/DEMODATA",
            output_path="build/",
        )
        context["plot_grid_t1"] = t1_plot
        logging.info("Added T1 plots to context")
    else:
        context["t1_plot_is_set"] = False
        pass

    path = Path("data/")

    ######### MERMIN TABLE
    maximum_mermin = fl.get_maximum_mermin(
        path / "mermin" / args.data_left, "results.json"
    )
    maximum_mermin_baseline = fl.get_maximum_mermin(
        path / "mermin" / args.data_right, "results.json"
    )
    context["mermin_maximum"] = maximum_mermin
    context["mermin_maximum_baseline"] = maximum_mermin_baseline

    ######### MERMIN PLOTS
    if args.mermin_5_plot == True:
        context["mermin_5_plot_is_set"] = True
        context["plot_mermin"] = pl.mermin_plot_5q(
            raw_data=os.path.join("data", "mermin", args.data_left, "results.json"),
            output_path="build/",
        )
        context["plot_mermin_baseline"] = mermin_5_plot_baseline = pl.mermin_plot_5q(
            raw_data=os.path.join("data", "mermin", args.data_right, "results.json"),
            output_path="build/",
        )
        logging.info("Added Mermin 5Q plots to context")
    else:
        context["mermin_5_plot_is_set"] = False
        pass

    ######### REUPLOADING PLOTS
    if args.reuploading_plot == True:
        context["reuploading_plot_is_set"] = True
        context["plot_reuploading"] = pl.do_plot_reuploading(
            raw_data=os.path.join("data", "reuploading", args.data_left, "results.json")
        )
        context["plot_reuploading_baseline"] = pl.do_plot_reuploading(
            raw_data=os.path.join(
                "data", "reuploading", args.data_right, "results.json"
            )
        )
        logging.info("Added reuploading plots to context")
    else:
        context["reuploading_plot_is_set"] = False
        pass

    ######### GROVER PLOTS
    if args.grover_plot == True:
        context["grover_plot_is_set"] = True
        context["plot_grover2q"] = pl.plot_grover(
            raw_data=os.path.join("data", "grover2q", args.data_left, "results.json"),
            output_path="build/",
        )
        context["plot_grover2q_baseline"] = pl.plot_grover(
            raw_data=os.path.join("data", "grover2q", args.data_right, "results.json"),
            output_path="build/",
        )
        logging.info("Added Grover 2Q plots to context")
    else:
        context["grover_plot_is_set"] = False
        pass

    ######### GHZ PLOTS
    if args.ghz_plot == True:
        context["ghz_plot_is_set"] = True
        context["plot_ghz"] = pl.plot_ghz(
            raw_data=os.path.join("data", "ghz", args.data_left, "results.json"),
            output_path="build/",
        )
        context["plot_ghz_baseline"] = pl.plot_ghz(
            raw_data=os.path.join("data", "ghz", args.data_right, "results.json"),
            output_path="build/",
        )
        logging.info("Added GHZ plots to context")
    else:
        context["ghz_plot_is_set"] = False
        pass

    return context


def render_and_save_report(context, args):
    """
    Render the LaTeX template and save the generated report.

    Args:
        context: Template context dictionary.
        args: Parsed command line arguments.

    Returns:
        Path to the generated LaTeX file.
    """
    logging.info("Rendering LaTeX template...")
    # Setup Jinja2 environment and load template
    template_dir = Path("src/templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.j2")

    # Render template with context
    rendered_content = template.render(context)

    # Ensure build directory exists
    build_dir = Path(".")
    build_dir.mkdir(exist_ok=True)

    # Write rendered LaTeX file in the build directory using os.path.join
    output_file = os.path.join(build_dir, "report.tex")
    with open(output_file, "w") as f:
        f.write(rendered_content)
        return output_file

    raise RuntimeError("Failed to render the report template.")


def main():
    """
    Main function that orchestrates the quantum benchmark report generation process.
    """
    logging.info("Starting report generation process...")

    # Step 1: Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Step 2: Load and validate experiment data (dummy call for now, can be removed if not needed for experiment_name)
    # For simplicity, we are not loading actual experiment data as per the request
    # to only show library versions.
    # If experiment_left is purely for the title, this can be simplified further.
    logging.info(
        f"Targeting experiment directory for report title: {args.experiment_left}"
    )

    # Step 3: Prepare template context with all required data
    context = prepare_template_context(args)

    # Step 4: Render and save the LaTeX report
    output_file = render_and_save_report(context, args)

    # Step 5: Report completion
    logging.info(f"Report generated at: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
