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
        "--experiment-dir",
        type=str,
        default="rb-1306",
        help="Directory containing the experiment data.",
    )
    parser.add_argument(
        "--experiment-dir-baseline",
        type=str,
        default="BASELINE",
        help="Directory containing the experiment data.",
    )
    return parser


def do_plot_reuploading(results_file):
    """
    Generate reuploading plots for each epoch and a final summary plot using data from the results JSON file.

    Args:
        results_file (str): Path to the JSON file containing reuploading results.
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    output_dir = os.path.dirname(results_file)

    # Generate a plot for each epoch
    for epoch_data in results["epoch_data"]:
        epoch = epoch_data["epoch"]
        x_train = np.array(epoch_data["x_train"])
        y_train = np.array(epoch_data["y_train"])
        predictions = np.array(epoch_data["predictions"])

        pl.plot_reuploading(
            x=x_train,
            target=y_train,
            predictions=predictions,
            title=f"epoch_{epoch:03d}",
            outdir=output_dir,
        )

    # Generate the final summary plot
    x_train = np.array(results["epoch_data"][-1]["x_train"])
    y_train = np.array(results["epoch_data"][-1]["y_train"])
    median_pred = np.array(results["median_predictions"])
    mad_pred = np.array(results["mad_predictions"])

    pl.plot_reuploading(
        x=x_train,
        target=y_train,
        predictions=median_pred,
        err=mad_pred,
        title="final_plot",
        outdir=output_dir,
    )

    return os.path.join(output_dir, "final_plot.pdf")


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
    meta_json_path = Path("data") / args.experiment_dir / "meta.json"
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
    logging.info("Loaded experiment metadata from %s", meta_json_path)

    # Fidelity statistics and changes
    stat_fidelity = fl.get_stat_fidelity(args.experiment_dir)
    stat_fidelity_baseline = fl.get_stat_fidelity(args.experiment_dir_baseline)
    stat_fidelity_with_improvement = add_stat_changes(
        stat_fidelity, stat_fidelity_baseline
    )
    logging.info("Prepared stat_fidelity and stat_fidelity_with_improvement")

    # Pulse Fidelity statistics and changes
    stat_pulse_fidelity = fl.get_stat_pulse_fidelity(args.experiment_dir)
    stat_pulse_fidelity_baseline = fl.get_stat_pulse_fidelity(
        args.experiment_dir_baseline
    )
    stat_pulse_fidelity_with_improvement = add_stat_changes(
        stat_pulse_fidelity, stat_pulse_fidelity_baseline
    )
    logging.info(
        "Prepared stat_pulse_fidelity and stat_pulse_fidelity_with_improvement"
    )

    # T1 statistics and changes
    stat_t1 = fl.get_stat_t12(args.experiment_dir, "t1")
    stat_t1_baseline = fl.get_stat_t12(args.experiment_dir_baseline, "t1")
    stat_t1_with_improvement = add_stat_changes(stat_t1, stat_t1_baseline)
    logging.info("Prepared stat_t1 and stat_t1_with_improvement")

    # T2 statistics and changes
    stat_t2 = fl.get_stat_t12(args.experiment_dir, "t2")
    stat_t2_baseline = fl.get_stat_t12(args.experiment_dir_baseline, "t2")
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
        "new_fidelity": fl.context_fidelity(args.experiment_dir),
        "control_fidelity": fl.context_fidelity(args.experiment_dir_baseline),
        #
        "plot_exp": pl.plot_fidelity_graph(
            args.experiment_dir, config.connectivity, config.pos
        ),
        "plot_baseline": pl.plot_fidelity_graph(
            args.experiment_dir_baseline, config.connectivity, config.pos
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

    # Add additional plots if needed
    context["grid_coupler_is_set"] = False
    grid_coupler_plots = pl.prepare_grid_coupler(
        max_number=2,
        data_dir="data/DEMODATA",
        baseline_dir="data/DEMODATA",
        output_path="build/",
    )
    context["plot_grid_coupler"] = grid_coupler_plots
    logging.info("Added grid_coupler plots to context")

    # Add additional chevron_swap_coupler plots if needed
    context["chevron_swap_coupler_is_set"] = False
    chevron_swap_coupler_plots = pl.prepare_grid_chevron_swap_coupler(
        max_number=2,
        data_dir="data/DEMODATA",
        baseline_dir="data/DEMODATA",
        output_path="build/",
    )
    context["plot_chevron_swap_coupler"] = chevron_swap_coupler_plots
    logging.info("Added chevron_swap_coupler plots to context")

    context["t1_plot_is_set"] = True
    t1_plot = pl.prepare_grid_t1_plts(
        max_number=2,
        data_dir="data/DEMODATA",
        output_path="build/",
    )
    context["plot_grid_t1"] = t1_plot
    logging.info("Added T1 plots to context")

    # MERMIN TABLE
    maximum_mermin = fl.get_maximum_mermin("data/mermin", "mermin_5q.json")
    maximum_mermin_baseline = fl.get_maximum_mermin("data/mermin", "mermin_5q.json")
    context["mermin_maximum"] = maximum_mermin
    context["mermin_maximum_baseline"] = maximum_mermin_baseline

    # MERMIN PLOTS
    context["mermin_5_plot_is_set"] = True
    mermin_5_plot_baseline = pl.mermin_plot_5q(
        raw_data="data/mermin/mermin_5q_baseline.json",
        output_path="build/",
    )
    mermin_5_plot = pl.mermin_plot_5q(
        raw_data="data/mermin/mermin_5q.json",
        output_path="build/",
    )
    context["plot_mermin_baseline"] = mermin_5_plot_baseline
    context["plot_mermin"] = mermin_5_plot
    logging.info("Added Mermin 5Q plots to context")

    # REUPLOADING PLOTS
    context["reuploading_plot_is_set"] = True
    context["plot_reuploading"] = do_plot_reuploading(
        "data/reuploading/results_reuploading.json"
    )
    context["plot_reuploading_baseline"] = do_plot_reuploading(
        "data/reuploading/results_reuploading.json"
    )
    logging.info("Added reuploading plots to context")

    # GROVER PLOTS
    context["grover_plot_is_set"] = True
    context["plot_grover2q"] = pl.plot_grover(
        "data/grover2q/results.json",
        output_path="build/",
    )
    context["plot_grover2q_baseline"] = pl.plot_grover(
        "data/grover2q/results.json",
        output_path="build/",
    )
    logging.info("Added Grover 2Q plots to context")

    # GHZ PLOTS
    context["ghz_plot_is_set"] = True
    context["plot_ghz"] = pl.plot_ghz(
        "data/ghz/ghz_5q_samples.json",
        output_path="build/",
    )
    context["plot_ghz_baseline"] = pl.plot_ghz(
        "data/ghz/ghz_5q_samples.json",
        output_path="build/",
    )
    logging.info("Added GHZ plots to context")

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
    # If experiment_dir is purely for the title, this can be simplified further.
    logging.info(
        f"Targeting experiment directory for report title: {args.experiment_dir}"
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
