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
                return f"(+{percent:.2f}\%)"
            elif percent < 0:
                return f"(-{percent:.2f}\%)"
            else:
                return "-"
            # return f"{diff:+.4g} ({percent:+.2f}\%)"
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
    print(f"Stat fidelity changes: {result}")
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

    # Fidelity statistics and changes
    stat_fidelity = fl.get_stat_fidelity(args.experiment_dir)
    stat_fidelity_baseline = fl.get_stat_fidelity(args.experiment_dir_baseline)
    stat_fidelity_with_improvement = add_stat_changes(
        stat_fidelity, stat_fidelity_baseline
    )

    # Pulse Fidelity statistics and changes
    stat_pulse_fidelity = fl.get_stat_pulse_fidelity(args.experiment_dir)
    stat_pulse_fidelity_baseline = fl.get_stat_pulse_fidelity(
        args.experiment_dir_baseline
    )
    stat_pulse_fidelity_with_improvement = add_stat_changes(
        stat_pulse_fidelity, stat_pulse_fidelity_baseline
    )

    # T1 statistics and changes
    stat_t1 = fl.get_stat_t12(args.experiment_dir, "t1")
    stat_t1_baseline = fl.get_stat_t12(args.experiment_dir_baseline, "t1")
    stat_t1_with_improvement = add_stat_changes(stat_t1, stat_t1_baseline)

    # T2 statistics and changes
    stat_t2 = fl.get_stat_t12(args.experiment_dir, "t2")
    stat_t2_baseline = fl.get_stat_t12(args.experiment_dir_baseline, "t2")
    stat_t2_with_improvement = add_stat_changes(stat_t2, stat_t2_baseline)

    context = {
        "experiment_name": meta_data.get("title", "Unknown Title"),
        "platform": meta_data.get("platform", "Unknown Platform"),
        #
        "start_time": meta_data.get("start-time", "Unknown Start Time"),
        "end_time": meta_data.get("start-time", "Unknown Start Time"),
        #
        "report_of_changes": "More report of changes (from software).",
        #
        # "stat_fidelity": fl.get_stat_fidelity(args.experiment_dir),
        # "stat_fidelity_baseline": fl.get_stat_fidelity(args.experiment_dir_baseline),
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
    }

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
