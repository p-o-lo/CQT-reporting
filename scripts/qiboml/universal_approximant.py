import os
import argparse
import pathlib
import json
from timeit import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from qibo.config import log
from qibo import Circuit, gates, construct_backend
from qibo.transpiler import (
    NativeGates,
    Passes,
    Unroller
)

from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR

os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

# settings
#backend = construct_backend("qibolab", platform="sinq-20")
backend = construct_backend("numpy") # for testing
nqubits = 1
qubit_id = 9
nlayers = 3
lr=0.25
epochs = 30
nshots = 500
num_samples = 15

# Prepare the training dataset 
def f(x):
    return 1 * torch.sin(x)  ** 2 - 0.3 * torch.cos(x) 

x_train = torch.linspace(0, 2 * np.pi, num_samples, dtype=torch.float64).unsqueeze(1)
y_train = f(x_train)
y_train = 2 * (((y_train - y_train.min()) / (y_train.max() - y_train.min())) - 0.5)

# Plotting functions
def plot_target(x, target, predictions=None, err=None, title="plot", outdir="."):
    """Plot target function and, optionally, the predictions of our model."""
    # flatten everything to 1D
    x = np.asarray(x).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if predictions is not None:
        predictions = np.asarray(predictions).reshape(-1)
    if err is not None:
        err = np.asarray(err).reshape(-1)

    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    plt.plot(x, target, marker="o", markersize=7, alpha=1, label="Targets", ls="-", markeredgecolor="black", color="red")
    if predictions is not None:
        plt.plot(x, predictions, marker="o", markersize=7, alpha=1, label="Predictions", ls="-", markeredgecolor="black", color="blue")
    if predictions is not None and err is not None:
        plt.fill_between(x, predictions - err, predictions + err, alpha=0.3, color="blue")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{title}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_loss_history(x, y, title="loss_plot", outdir=".", show=False):
    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    plt.plot(x, y)
    plt.xlabel(r"$Iteration$")
    plt.ylabel(r"$Loss$")
    os.makedirs(outdir, exist_ok=True)
    if show:
        plt.show()
    plt.savefig(os.path.join(outdir, f"{title}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

# Trainable layer
def trainable_circuit(nqubits, entanglement=True, density_matrix=False):
    """
    Construct a trainable quantum circuit where the amount of entanglement can be tuned 
    if the argument `entanglement` is set equal to `True`."""
    circ = Circuit(nqubits, density_matrix=density_matrix)
    for q in range(nqubits):
        circ.add(gates.RY(q=q, theta=np.random.randn()))
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
    if nqubits > 1 and entanglement:
        for q in range(nqubits):
            circ.add(gates.CRX(q, (q+1)%nqubits, theta=np.random.randn()))
    return circ

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small quantum model on a sin^2 function."
    )
    args = parser.parse_args()

    # and put the transpiler here
    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)

    transpiler = custom_pipeline

    backend_name = backend.name.lower().replace(" ", "_")
    results_dir = os.path.join("results", f"universal_approx_{backend_name}_{nshots}shots")
    params_dir = os.path.join(results_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # Build circuits
    encoding_circ = PhaseEncoding(nqubits=nqubits, encoding_gate=gates.RX, density_matrix=False)
    circuit_structure = []
    for _ in range(nlayers):
        circuit_structure.extend([encoding_circ, trainable_circuit(nqubits, entanglement=False, density_matrix=False)])

    decoding_circ = Expectation(
        nqubits=nqubits,
        nshots=nshots,
        backend=backend,
        transpiler=transpiler,
        mitigation_config=None,
        noise_model=None,
        density_matrix=False,
        wire_names=[qubit_id]
    )

    # Model
    model_kwargs = dict(circuit_structure=circuit_structure, decoding=decoding_circ)
    model_kwargs["differentiation"] = PSR()
    model = QuantumModel(**model_kwargs)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    loss_history = []
    time_start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = torch.stack([model(x) for x in x_train]).squeeze(-1)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        torch.save(model.state_dict(),
                   os.path.join(params_dir, f"epoch_{epoch:03d}.pt"))

        log.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        plot_target(
            x_train.detach().numpy(),
            y_train.detach().numpy(),
            predictions=y_pred.detach().numpy(),
            err=None,
            title=f"fit_epoch_{epoch:03d}",
            outdir=results_dir
        )
    time_end = time.time()
    duration = time_end - time_start

    np.save(os.path.join(results_dir, "loss_history.npy"), np.array(loss_history))

    preds = np.stack([
        torch.stack([model(x) for x in x_train]).squeeze(-1).detach().numpy()
        for _ in range(20)
    ], axis=0)
    median_pred = np.median(preds, axis=0)
    mad_pred = np.median(np.abs(preds - median_pred[None, :]), axis=0)

    np.save(os.path.join(results_dir, "median_predictions.npy"), median_pred)
    np.save(os.path.join(results_dir, "mad_predictions.npy"), mad_pred)

    plot_target(
        x_train.detach().numpy(),
        y_train.detach().numpy(),
        predictions=median_pred,
        err=mad_pred,
        title="final_plot",
        outdir=results_dir
    )
    
    plot_loss_history(
        range(len(loss_history)),
        loss_history,
        title="loss_history_plot",
        outdir=results_dir
    ) 

    report_data = {
        "final_loss": loss_history[-1],
        "duration": duration,
        "x_train": x_train.detach().numpy().tolist(),
        "y_train": y_train.detach().numpy().tolist(),
        "predictions": median_pred.tolist(),
        "err": mad_pred.tolist(),
    }
    
    with open(os.path.join(results_dir, f"report_data.json"), "w") as file:
        json.dump(report_data, file, indent=4)

