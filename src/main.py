import argparse
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import logging
import os
import json

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


def context_new_version(args, meta_data):
    """
    Get the current version of the libraries used in the benchmarking suite.
    """
    return {
        "versions": meta_data.get("versions", {}),
        "runcard": meta_data.get("runcard", "What do you want here?"),
        "runcard_link": meta_data.get("runcard_link", "https://link-to-runcard.com"),
    }


def context_control_version(args):
    """
    Get the control version of the libraries used in the benchmarking suite.
    """
    meta_json_path = Path("data") / args.experiment_dir_baseline / "meta.json"
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)
    return {
        "versions": meta_data.get("versions", {}),
        "runcard": meta_data.get("runcard", "What do you want here?"),
        "runcard_link": meta_data.get("runcard_link", "https://link-to-runcard.com"),
    }


def context_fidelity(experiment_dir):
    """
    Extracts the list of fidelities and error bars from the experiment results.
    Returns a list of dicts: {"fidelity": ..., "error_bars": ...}
    """
    results_json_path = Path("data") / experiment_dir / "data/rb-0/results.json"
    with open(results_json_path, "r") as f:
        results = json.load(f)

    fidelities = results.get('"fidelity"', {})
    error_bars = results.get('"error_bars"', {})

    # Convert dict_values to list and get the first item (which should be a list)
    fidelities_list = list(fidelities.values())
    # Extract the first element from each key ("1", "2", ..., "20") in error_bars
    error_bars_list = []
    for k in sorted(error_bars.keys(), key=lambda x: int(x)):
        values = error_bars[k]
        if isinstance(values, list) and values:
            error_bars_list.append(values[0])
        else:
            error_bars_list.append(None)

    _ = [
        {
            "qn": i,
            "fidelity": f"{f:.4g}" if isinstance(f, (float, int)) else f,
            "error_bars": f"{e:.4g}" if isinstance(e, (float, int)) else e,
        }
        for i, (f, e) in enumerate(zip(fidelities_list, error_bars_list))
    ]
    # Debugging line to inspect the fidelity and error bars
    # Zip and return as list of dicts for template clarity
    return _


def prepare_template_context(args):
    """
    Prepare a complete template context for the benchmarking report.
    """
    logging.info("Preparing context for full benchmarking report.")

    # Load experiment metadata from mega.json

    meta_json_path = Path("data") / args.experiment_dir / "meta.json"
    with open(meta_json_path, "r") as f:
        meta_data = json.load(f)

    context = {
        "experiment_name": meta_data.get("title", "Unknown Title"),
        "platform": meta_data.get("platform", "Unknown Platform"),
        #
        "start_time": meta_data.get("start-time", "Unknown Start Time"),
        "end_time": meta_data.get("start-time", "Unknown Start Time"),
        #
        "report_of_changes": "More report of changes (from software).",
        #
        "new_version": context_new_version(args, meta_data),
        "control_version": context_control_version(args),
        #
        "new_fidelity": context_fidelity(args.experiment_dir),
        "control_fidelity": context_fidelity(args.experiment_dir_baseline),
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
