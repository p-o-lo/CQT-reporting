import os
import torch
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # scripts/config.py


import json


torch.set_default_dtype(torch.float64)
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn

from qibo.noise import PauliError, NoiseModel
from qibo.config import log
from qibo import Circuit, gates, construct_backend
from qibo.transpiler import NativeGates, Passes, Unroller

from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR

# os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()


# Prepare the training dataset
def f(x):
    return 1 * torch.sin(x) ** 2 - 0.3 * torch.cos(x)


def build_noise_model(nqubits: int, local_pauli_noise_prob: float):
    """Construct a local Pauli noise channel + readout noise model."""
    noise_model = NoiseModel()
    for q in range(nqubits):
        noise_model.add(
            PauliError(
                [
                    ("X", local_pauli_noise_prob),
                    ("Y", local_pauli_noise_prob),
                    ("Z", local_pauli_noise_prob),
                ]
            ),
            qubits=q,
        )
    return noise_model


# # === FIXED plotting function ===
# def plot_target(x, target, predictions=None, err=None, title="plot", outdir="."):
#     """Plot target function and, optionally, the predictions of our model."""
#     # flatten everything to 1D
#     x = np.asarray(x).reshape(-1)
#     target = np.asarray(target).reshape(-1)
#     if predictions is not None:
#         predictions = np.asarray(predictions).reshape(-1)
#     if err is not None:
#         err = np.asarray(err).reshape(-1)

#     plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
#     plt.plot(
#         x,
#         target,
#         marker="o",
#         markersize=7,
#         alpha=1,
#         label="Targets",
#         ls="-",
#         markeredgecolor="black",
#         color="red",
#     )
#     if predictions is not None:
#         plt.plot(
#             x,
#             predictions,
#             marker="o",
#             markersize=7,
#             alpha=1,
#             label="Predictions",
#             ls="-",
#             markeredgecolor="black",
#             color="blue",
#         )
#     if predictions is not None and err is not None:
#         plt.fill_between(
#             x, predictions - err, predictions + err, alpha=0.3, color="blue"
#         )
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$f(x)$")
#     plt.legend()
#     os.makedirs(outdir, exist_ok=True)
#     plt.savefig(os.path.join(outdir, f"{title}.pdf"), dpi=300, bbox_inches="tight")
#     plt.close()


# # === end fix ===


# Trainable layer
def trainable_circuit(entanglement=True, density_matrix=False):
    """
    Construct a trainable quantum circuit where the amount of entanglement can be tuned
    if the argument `entanglement` is set equal to `True`."""
    circ = Circuit(nqubits, density_matrix=density_matrix)
    for q in range(nqubits):
        circ.add(gates.RY(q=q, theta=np.random.randn()))
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
    if nqubits > 1 and entanglement:
        for q in range(nqubits):
            circ.add(gates.CRX(q, (q + 1) % nqubits, theta=np.random.randn()))
    return circ


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small quantum model on a sin^2 function."
    )
    parser.add_argument(
        "--with_mitigation",
        action="store_true",
        help="If set, use error mitigation (CDR) with local Pauli noise.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="Number of training samples (default: 15)",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        default=3,
        help="Number of layers in the quantum circuit (default: 3)",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20"],
        default="numpy",
        help="Execution device (forwarded by runscripts; unused here).",
    )
    args = parser.parse_args()

    x_train = torch.linspace(
        0, 2 * np.pi, args.num_samples, dtype=torch.float64
    ).unsqueeze(1)

    y_train = f(x_train)
    y_train = 2 * (((y_train - y_train.min()) / (y_train.max() - y_train.min())) - 0.5)

    # Configuration
    nqubits = 1
    nshots = 500

    backend = construct_backend("numpy")  # , platform="sinq-20")

    # and put the transpiler here
    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)

    transpiler = custom_pipeline

    backend_name = backend.name.lower().replace(" ", "_")
    tag = f"rtqem{args.with_mitigation}"
    # results_dir = os.path.join(
    #    "data/reuploading/results", f"reuploading_{backend_name}_{nshots}shots_{tag}"
    # )
    results_dir = (config.output_dir_for(__file__) / args.device).as_posix()
    params_dir = os.path.join(results_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # Noise & mitigation
    noise_model = (
        build_noise_model(nqubits, 0.02) if "qibolab" not in backend_name else None
    )
    mitigation_config = (
        {
            "threshold": 1e-1,
            "method": "CDR",
            "min_iterations": 500,
            "method_kwargs": {"n_training_samples": 100, "nshots": 10000},
        }
        if args.with_mitigation
        else None
    )
    density_matrix = True if noise_model is not None else False

    log.info(f"Mitigation config: {mitigation_config}, noise model: {noise_model}")

    # Build circuits
    encoding_circ = PhaseEncoding(
        nqubits=nqubits, encoding_gate=gates.RX, density_matrix=density_matrix
    )
    circuit_structure = []
    for _ in range(args.nlayers):
        circuit_structure.extend(
            [
                encoding_circ,
                trainable_circuit(entanglement=False, density_matrix=density_matrix),
            ]
        )

    decoding_circ = Expectation(
        nqubits=nqubits,
        nshots=nshots,
        backend=backend,
        transpiler=transpiler,
        # mitigation_config=mitigation_config,
        # noise_model=noise_model,
        # density_matrix=density_matrix,
        # wire_names=[9],
    )

    # Model
    model_kwargs = dict(circuit_structure=circuit_structure, decoding=decoding_circ)
    if args.with_mitigation:
        model_kwargs["differentiation"] = PSR()
    model = QuantumModel(**model_kwargs)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=0.25)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 30
    loss_history = []
    plot_data = []  # To store data for all epochs

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = torch.stack([model(x) for x in x_train]).squeeze(-1)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        torch.save(
            model.state_dict(), os.path.join(params_dir, f"epoch_{epoch:03d}.pt")
        )

        log.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        # Collect data for JSON
        epoch_data = {
            "epoch": epoch,
            "x_train": x_train.detach().numpy().tolist(),
            "y_train": y_train.detach().numpy().tolist(),
            "predictions": y_pred.detach().numpy().tolist(),
            "loss": loss.item(),
        }
        plot_data.append(epoch_data)

    # # Save all epoch data to a JSON file
    # with open(os.path.join(results_dir, "epoch.json"), "w") as json_file:
    #     json.dump(plot_data, json_file, indent=4)

    # # Save loss history to a JSON file
    # with open(os.path.join(results_dir, "loss_history.json"), "w") as json_file:
    #     json.dump({"loss_history": loss_history}, json_file, indent=4)

    preds = np.stack(
        [
            torch.stack([model(x) for x in x_train]).squeeze(-1).detach().numpy()
            for _ in range(20)
        ],
        axis=0,
    )
    median_pred = np.median(preds, axis=0)
    mad_pred = np.median(np.abs(preds - median_pred[None, :]), axis=0)

    # Save predictions and errors
    results = {
        "epoch_data": plot_data,
        "loss_history": loss_history,
        "median_predictions": median_pred.tolist(),
        "mad_predictions": mad_pred.tolist(),
    }
    with open(os.path.join(results_dir, "results.json"), "w") as json_file:
        json.dump(results, json_file, indent=4)
