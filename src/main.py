import argparse
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import logging

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
    # Removed other arguments as they are not used in the simplified version
    return parser


def prepare_template_context(args):
    """
    Prepare a complete template context for the benchmarking report.
    """
    logging.info("Preparing context for full benchmarking report.")

    context = {
        "chip_name": "chip123",
        "qubits": "5",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "report_of_changes": "Initial benchmarking suite run.",
        "new_version": {
            "packages": [
                {"library": "qibo", "version": "0.2.18"},
                {"library": "numpy", "version": "2.3.0"},
                {"library": "qibolab", "version": "0.2.7"},
                {"library": "qibocal", "version": "0.2.2"},
            ],
            "runcard": "runcard_v2.yml",
            "runcard_link": "https://example.com/runcard_v2.yml",
        },
        "control_version": {
            "packages": [
                {"library": "qibo", "version": "0.2.17"},
                {"library": "numpy", "version": "2.2.0"},
                {"library": "qibolab", "version": "0.2.6"},
                {"library": "qibocal", "version": "0.2.1"},
            ],
            "runcard": "runcard_v1.yml",
            "runcard_link": "https://example.com/runcard_v1.yml",
        },
        "routine_results": [
            {
                "routine": "RB",
                "avg_result_error": "0.95±0.01",
                "best_qubit": "Q1",
                "worst_qubit": "Q5",
            },
            {
                "routine": "T1",
                "avg_result_error": "100±5",
                "best_qubit": "Q2",
                "worst_qubit": "Q4",
            },
        ],
        "t1t2_values": [
            {"qubit": "Q1", "t1": "110", "t2": "90", "outlier": "No"},
            {"qubit": "Q2", "t1": "120", "t2": "95", "outlier": "No"},
            {"qubit": "Q3", "t1": "80", "t2": "60", "outlier": "Yes"},
        ],
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
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)

    # Write rendered LaTeX file in the build directory
    output_file = build_dir / "report.tex"
    with open(output_file, "w") as f:
        f.write(rendered_content)

    logging.info(f"Report saved to: {output_file}")
    return output_file


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
    logging.info(f"Report generated: {output_file}")
    logging.info(
        f"To compile to PDF, run: cd build && pdflatex {output_file.name} && mv report.pdf ../"
    )

    return 0


if __name__ == "__main__":
    exit(main())
