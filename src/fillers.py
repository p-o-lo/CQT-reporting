import argparse
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import logging
import os
import json
import numpy as np
import pdb


import matplotlib.pyplot as plt


def context_plot_1(exp_name):
    """
    Generates a plot with y-axis from 0 to 1 and x-axis from 0 to 500.
    Returns the filepath of the saved image.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Plot 1: what do you want?")
    # Optionally, plot a line or leave empty
    # ax.plot([], [])
    output_dir = Path("build")
    output_dir.mkdir(exist_ok=True)
    filename = f"plot_1_{exp_name}.pdf"
    filepath = output_dir / filename
    plt.savefig(filepath)
    plt.close(fig)
    return str(filepath)


def context_new_version(args, meta_data):
    """
    Get the current version of the libraries used in the benchmarking suite.
    """
    return {
        "versions": meta_data.get("versions", {}),
        "runcard": meta_data.get("runcard", args.experiment_dir),
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
        "runcard": meta_data.get("runcard", args.experiment_dir_baseline),
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
            "fidelity": f"{f:.3g}" if isinstance(f, (float, int)) else f,
            "error_bars": f"{e:.3g}" if isinstance(e, (float, int)) else e,
        }
        for i, (f, e) in enumerate(zip(fidelities_list, error_bars_list))
    ]
    # mark the best fidelity by writing the element to \textcolor{green}{element}
    max_fidelity = np.nanmax(
        [
            float(item["fidelity"])
            for item in _
            if isinstance(item["fidelity"], (float, int))
            or (
                isinstance(item["fidelity"], str)
                and item["fidelity"].replace(".", "", 1).isdigit()
            )
        ]
    )
    # print(max_fidelity)
    _ = [
        {
            "qn": item["qn"],
            "fidelity": (
                f"\\textcolor{{green}}{{{item['fidelity']}}}"
                if (
                    np.isclose(
                        float(item["fidelity"]), max_fidelity, rtol=1e-3, atol=1e-3
                    )
                )
                else item["fidelity"]
            ),
            "error_bars": item["error_bars"],
        }
        for item in _
    ]

    # Debugging line to inspect the fidelity and error bars
    # Zip and return as list of dicts for template clarity
    return _


def get_stat_fidelity(experiment_dir):
    """
    Returns a dictionary with average, min, max, and median fidelity for the given experiment directory.
    """
    results_json_path = Path("data") / experiment_dir / "data/rb-0/results.json"
    with open(results_json_path, "r") as f:
        results = json.load(f)

    fidelities = results.get('"fidelity"', {})
    # Convert dict_values to list and flatten if needed
    fidelities_list = list(fidelities.values())

    # Convert all values to float if possible
    numeric_fidelities = []
    for f in fidelities_list:
        try:
            numeric_fidelities.append(float(f))
        except (ValueError, TypeError):
            continue

    if not numeric_fidelities:
        return {"average": None, "min": None, "max": None, "median": None}

    dict_fidelities = {
        "average": f"{np.nanmean(numeric_fidelities):.3g}",
        "min": f"{np.nanmin(numeric_fidelities):.3g}",
        "max": f"{np.nanmax(numeric_fidelities):.3g}",
        "median": f"{np.nanmedian(numeric_fidelities):.3g}",
    }

    return dict_fidelities


def get_stat_t12(experiment_dir, stat_type):
    """
    Returns a dictionary with average, min, max, and median T1 for the given experiment directory.
    """
    results_json_path = Path("data") / experiment_dir / "platform/calibration.json"
    with open(results_json_path, "r") as f:
        results = json.load(f)

    # Extract T1 values for all qubits from the "single_qubits" section
    single_qubits = results.get("single_qubits", {})
    ts_list = []
    for qubit_data in single_qubits.values():
        t1_values = qubit_data.get(stat_type, [])
        # Only consider non-null, numeric values
        for t in t1_values:
            if t is not None:
                try:
                    ts_list.append(float(t))
                except (ValueError, TypeError):
                    continue

    numeric_ts = ts_list

    if not numeric_ts:
        return {"average": None, "min": None, "max": None, "median": None}

    dict_ts = {
        "average": f"{np.nanmean(numeric_ts):.3g}",
        "min": f"{np.nanmin(numeric_ts):.3g}",
        "max": f"{np.nanmax(numeric_ts):.3g}",
        "median": f"{np.nanmedian(numeric_ts):.3g}",
    }
    return dict_ts


def get_stat_pulse_fidelity(experiment_dir):
    """
    Returns a dictionary with average, min, max, and median pulse fidelity for the given experiment directory.
    """
    results_json_path = Path("data") / experiment_dir / "data/rb-0/results.json"
    with open(results_json_path, "r") as f:
        results = json.load(f)

    pulse_fidelities = results.get('"pulse_fidelity"', {})
    # Convert dict_values to list and flatten if needed
    pulse_fidelities_list = list(pulse_fidelities.values())

    # Convert all values to float if possible
    numeric_pulse_fidelities = []
    for f in pulse_fidelities_list:
        try:
            numeric_pulse_fidelities.append(float(f))
        except (ValueError, TypeError):
            continue

    if not numeric_pulse_fidelities:
        return {"average": None, "min": None, "max": None, "median": None}

    dict_pulse_fidelities = {
        "average": f"{np.nanmean(numeric_pulse_fidelities):.3g}",
        "min": f"{np.nanmin(numeric_pulse_fidelities):.3g}",
        "max": f"{np.nanmax(numeric_pulse_fidelities):.3g}",
        "median": f"{np.nanmedian(numeric_pulse_fidelities):.3g}",
    }

    return dict_pulse_fidelities
