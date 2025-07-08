import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import json
import os


def prepare_grid_coupler(
    max_number,
    data_dir="data/DEMODATA",
    baseline_dir="data/DEMODATA",
    output_path="build/",
):
    """
    Prepare the grid for the report.
    Returns a list of dicts with 'plot' and 'baseline' keys for Jinja template.
    """
    plot_grid = []
    for i in range(max_number):
        plot_path = plot_swap_coupler(
            qubit_number=i, data_dir=data_dir, output_path=output_path
        )
        baseline_path = plot_swap_coupler(
            qubit_number=i, data_dir=baseline_dir, output_path=output_path
        )
        plot_grid.append(
            {
                "plot": plot_path,
                "baseline": baseline_path,
            }
        )
    return plot_grid


def prepare_grid_chevron_swap_coupler(
    max_number,
    data_dir="data/DEMODATA",
    baseline_dir="data/DEMODATA",
    output_path="build/",
):
    """
    Prepare the grid for chevron swap coupler plots for the report.
    Returns a list of dicts with 'plot' and 'baseline' keys for Jinja template.
    """
    plot_grid = []
    for i in range(max_number):
        plot_path = plot_chevron_swap_coupler(
            qubit_number=i, data_dir=data_dir, output_path=output_path
        )
        baseline_path = plot_chevron_swap_coupler(
            qubit_number=i, data_dir=baseline_dir, output_path=output_path
        )
        plot_grid.append(
            {
                "plot": plot_path,
                "baseline": baseline_path,
            }
        )
    return plot_grid


def plot_fidelity_graph(
    experiment_name, connectivity, pos, output_path="build/", demo_data="data/DEMODATA/"
):
    """
    Generates a fidelity graph for the given experiment.
    """

    # temporary fix for demo data
    results_demo_json_path = Path(demo_data) / f"fidelity2qb_{experiment_name}.json"
    with open(results_demo_json_path, "r") as f:
        results_tmp = json.load(f)
    # this will be changed because of MLK - beware of stringescape issues
    fidelities_2qb = results_tmp.get('"fidelities_2qb"', {})

    # Load results for the main path
    results_json_path = "data" / Path(experiment_name) / "data/rb-0/results.json"
    with open(results_json_path, "r") as f:
        results = json.load(f)
    fidelities = results.get('"fidelity"', {})

    labels = {
        a: f"Q{a}\n{np.round(fidelities[str(a)] * 100, decimals=2)}" for a in range(20)
    }

    g = nx.Graph()
    cmap = plt.get_cmap("viridis")
    array = np.array(list(fidelities.values())) * 100
    node_color = [
        (
            plt.cm.viridis((value * 100 - min(array)) / (max(array) - min(array)))
            if value > 0.8
            else "grey"
        )
        for value in list(fidelities.values())
    ]
    levels = MaxNLocator(nbins=100).tick_values(min(array), max(array))
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    g.add_nodes_from(list(range(20)))
    g.add_edges_from(connectivity)
    nx.draw_networkx_edges(g, pos, edge_color="black", width=5)
    nx.draw_networkx_nodes(g, pos, node_size=800, linewidths=5, node_color=node_color)
    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        font_color="r",
        alpha=0.6,
        font_size=8,
        font_weight="bold",
    )
    # print(connectivity)
    # import pdb

    # pdb.set_trace()
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels={
            (a, b): (
                f"{np.round(fidelities_2qb[f'({a},{b})'] * 100, decimals=2)}"
                if f"({a},{b})" in fidelities_2qb
                else "-"
            )
            for a, b in connectivity
        },
        font_color="black",
        font_size=8,
        font_weight="bold",
    )

    ax = plt.gca()
    # Place colorbar below the plot
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        pad=0.1,
        fraction=0.05,
        aspect=40,  # Make the colorbar longer
        shrink=0.8,  # Adjust shrink to make it visually longer
    )
    cbar.set_label("1Q Fidelity")
    plt.box(False)
    plt.tight_layout()
    filename = "experiment_name.pdf"
    full_path = output_path + filename
    plt.savefig(full_path)
    plt.close()
    return full_path


def plot_swap_coupler(qubit_number=0, data_dir="data/DEMODATA", output_path="build/"):
    """
    Plots the swap coupler data for a given qubit index.

    Args:
        QUBIT (int): Qubit index (0-based).
        data_dir (str): Directory containing the JSON data file.
    """

    file_path = os.path.join(data_dir, f"swap_q{qubit_number + 1}_coupler.json")
    with open(file_path) as r:
        data = json.load(r)

    x = np.array(data["x"])
    y = np.array(data["y"])
    res = np.array(data["data"]).transpose() * 1e6

    fig, ax = plt.subplots()
    levels = MaxNLocator(nbins=100).tick_values(res.min(), res.max())
    cmap = plt.get_cmap("inferno")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    im = ax.imshow(
        res,
        cmap=cmap,
        norm=norm,
        extent=(x[0], x[-1], y[-1], y[0]),
        aspect="auto",
        origin="upper",
        interpolation="none",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transmission [ADC arb. units]")
    ax.axhline(0.12, label="Coupler Sweetspot", color="blue", linestyle="dashed")
    ax.set_ylabel("Coupler Amplitude [arb. units]")
    ax.set_xlabel("SWAP Length [ns]")
    fig.tight_layout()
    output_path = os.path.join(
        data_dir, f"chevron_swap_q{qubit_number + 1}_coupler.pdf"
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_chevron_swap_coupler(
    qubit_number=0, data_dir="data/DEMODATA/", output_path="build/"
):
    """
    Chevron plot for the SWAP coupler data.

    Args:
        qubit_number (int, optional): _description_. Defaults to 0.
        data_dir (str, optional): _description_. Defaults to "./data/".
        output_path (str, optional): _description_. Defaults to "build/".
    """

    # Use data_dir for the file path
    file_path = os.path.join(data_dir, f"swap_q{qubit_number + 1}.json")
    with open(file_path) as r:
        data = json.load(r)

    x = np.array(data["x"])
    y = np.array(data["y"])
    res = np.array(data["data"]).transpose() * 1e6

    fig, ax = plt.subplots()
    levels = MaxNLocator(nbins=100).tick_values(res.min(), res.max())
    cmap = plt.get_cmap("inferno")
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    im = ax.imshow(
        res,
        cmap=cmap,
        norm=norm,
        extent=(x[0], x[-1], y[-1], y[0]),
        aspect="auto",
        origin="upper",
        interpolation="none",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transmission [ADC arb. units]")
    ax.set_ylabel("SWAP Amplitude [arb. units]")
    ax.set_xlabel("SWAP Length [ns]")
    ax.set_title(f"Q13")
    fig.tight_layout()
    output_path = os.path.join(output_path, f"chevron_swap_q{qubit_number + 1}.pdf")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def prepare_grid_t1_plts(max_number, data_dir, output_path="build/"):
    """
    Prepare a grid of dummy T1 decay plots for the report.
    Returns a list of dicts with 'plot' and 'plot_baseline' keys for Jinja template.
    """
    plot_grid = []
    for i in range(max_number):
        plot_path = plot_t1_decay(
            qubit_number=i, data_dir=data_dir, output_path=output_path
        )
        plot_baseline_path = plot_t1_decay(
            qubit_number=i,
            data_dir=data_dir,
            output_path=output_path,
            suffix="_baseline",
        )
        plot_grid.append({"plot": plot_path, "plot_baseline": plot_baseline_path})
    return plot_grid


def plot_t1_decay(qubit_number, data_dir, output_path="build/", suffix=""):
    """
    Create a dummy T1 decay plot: exponential decay from 1 to 0.
    X axis: milliseconds, Y axis: T1
    """

    x = np.linspace(0, 10, 100)
    y = np.exp(-x / 3)  # Arbitrary decay constant for mockup
    plt.figure()
    plt.plot(x, y, label=f"Qubit {qubit_number}")
    plt.xlabel("milliseconds")
    plt.ylabel("T1")
    plt.title(f"T1 Decay Qubit {qubit_number}{suffix}")
    plt.ylim(0, 1.05)
    plt.xlim(0, 10)
    plt.legend()
    plt.tight_layout()
    filename = f"t1_{qubit_number}{suffix}.pdf"
    full_path = os.path.join(output_path, filename)
    plt.savefig(full_path)
    plt.close()
    return full_path
