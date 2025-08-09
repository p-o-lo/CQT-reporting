import os
import argparse
import pathlib
import json
from timeit import time
import tqdm
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

from qibo.config import log
from qibo import Circuit, gates, construct_backend
from qibo.transpiler import (
    NativeGates,
    Passes,
    Unroller
)
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from qiboml.models.encoding import PhaseEncoding, QuantumEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR
from qiboml import ndarray

# os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path(
#     "/mnt/scratch/qibolab_platforms_nqch"
# ).as_posix()

# Prepare the training dataset 
def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1

    return points, labels

def create_dataset(grid=None, samples=1000, seed=0):
    """Function to create training and test sets for classifying.

    Args:
        name (str): Name of the problem to create the dataset, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
        grid (int): Number of points in one direction defining the grid of points.
            If not specified, the dataset does not follow a regular grid.
        samples (int): Number of points in the set, randomly located.
            This argument is ignored if grid is specified.
        seed (int): Random seed

    Returns:
        Dataset for the given problem (x, y)
    """
    if grid is None:
        np.random.seed(seed)
        points = 1 - 2 * np.random.rand(samples, 2)
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    creator = _circle

    x, y = creator(points)
    return x, y

# Plotting functions
def plot_predictions(x, y, title="scatter_plot", outdir=".", show=False):
    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    for label in np.unique(y):
        data_label = np.transpose(x[np.where(y==label)])
        plt.scatter(data_label[0], data_label[1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    circle = plt.Circle((0, 0), np.sqrt(2 / np.pi), edgecolor ='k', linestyle='--', fill=False)
    plt.gca().add_patch(circle)
    os.makedirs(outdir, exist_ok=True)
    if show:
        plt.show()
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
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
        circ.add(gates.RX(q=q, theta=np.pi/2, trainable=False))
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
        circ.add(gates.RX(q=q, theta=-np.pi/2, trainable=False))
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
    return circ

def predict(model, data):
    test_pred = torch.as_tensor([torch.sigmoid(model(x)) for x in data])
    test_pred = test_pred.to("cpu")
    test_pred_int = torch.round(test_pred)

    return test_pred_int.tolist()

def compute_accuracy(labels, predictions, tolerance=1e-2):
    """
    Args:
        labels: numpy.array with the labels of the quantum states to be classified.
        predictions: numpy.array with the predictions for the quantum states classified.
        tolerance: float tolerance level to consider a prediction correct (default=1e-2).
    Returns:
        float with the proportion of states classified successfully.
    """
    accur = 0
    for l, p in zip(labels, predictions):
        if np.allclose(l, p, rtol=0.0, atol=tolerance):
            accur += 1

    accur = accur / len(labels)

    return accur

# RxRy Encoding
@dataclass(eq=False)
class RYRZEncoding(QuantumEncoding):

    def __post_init__(
        self,
    ):
        """Ancillary post initialization: builds the internal circuit with the rotation gates."""
        super().__post_init__()

    @cached_property
    def _data_to_gate(self):
        """
        Mapping between the index of the input and the indices of the gates in the
        produced encoding circuit queue, where the input is encoded to.
        For instance, {0: [0,2], 1: [2]}, represents an encoding where the element
        0 of the inputs enters the gates with indices 0 and 2 of the queue, whereas
        the element 1 of the input affects only the the gate in position 2 of the
        queue.
        By deafult, the map reproduces a simple encoding where the
        i-th component of the data is uploaded in the i-th gate of the queue.
        """
        return {f"{i}": [i] for i in range(len(self.qubits)*2)}

    def __call__(self, x: ndarray) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the chosen encoding gate.

        Args:
            x (ndarray): the input real data to encode in rotation angles.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        circuit = self.circuit
        x = x.ravel()
        for i, q in enumerate(self.qubits):
            for j,gate in enumerate([gates.RY, gates.RZ]):
                this_gate_params = {"trainable": False}
                this_gate_params.update({"theta": x[2*i+j]})
                circuit.add(gate(q=q, **this_gate_params))
        return circuit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small quantum model on a classification task for the circle dataset."
    )
    parser.add_argument(
        "--backend", type=str, default="numpy", help="Backend to use (default: numpy)"
    )
    parser.add_argument(
        "--nqubits", type=int, default=1, help="Number of qubits (default: 1)"
    )
    parser.add_argument("--qubit_id", type=int, default=9, help="Qubit ID (default: 9)")
    parser.add_argument(
        "--nlayers", type=int, default=6, help="Number of layers (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--nshots", type=int, default=500, help="Number of shots (default: 500)"
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=11,
        help="Number of grid points along both paramter directions for generating training dataset (default: 11)",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=100,
        help="Number of test samples (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=99, help="Random seed value for initializing trainable circuit parameters (default: 99)"
    )
    args = parser.parse_args()

    # Generate training data
    training_set = create_dataset(grid=args.grid)
    test_set = create_dataset(samples=args.num_test_samples)
    x_train = torch.tensor(training_set[0])
    y_train = torch.tensor(training_set[1].reshape([training_set[1].shape[0],1]), dtype=torch.float64)
    x_test = torch.tensor(test_set[0])
    y_test = torch.tensor(test_set[1].reshape([test_set[1].shape[0],1]), dtype=torch.float64)

    # Set up backend
    backend = construct_backend(args.backend)

    # Set up transpiler
    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)
    transpiler = custom_pipeline

    # Set up directories
    backend_name = backend.name.lower().replace(" ", "_")
    results_dir = os.path.join("data", f"reuploading_classifier/")
    params_dir = os.path.join(results_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    
    # Build circuits
    encoding = RYRZEncoding(nqubits=args.nqubits, density_matrix=False)
    circuit_structure = []

    np.random.seed(args.seed)
    for _ in range(args.nlayers):
        circuit_structure.extend([encoding])
        circuit_structure.extend([trainable_circuit(args.nqubits, entanglement=False, density_matrix=False)])

    observable = SymbolicHamiltonian(Z(0), nqubits=args.nqubits)
    
    decoding_circ = Expectation(
        nqubits=args.nqubits,
        observable=observable,
        nshots=args.nshots,
        backend=backend,
        transpiler=transpiler,
        mitigation_config=None,
        noise_model=None,
        density_matrix=False,
        wire_names=[args.qubit_id]
    )

    # Model
    model_kwargs = dict(circuit_structure=circuit_structure, decoding=decoding_circ)
    model_kwargs["differentiation"] = PSR()
    model = QuantumModel(**model_kwargs)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.binary_cross_entropy_with_logits

    # Training loop
    loss_history = []
    time_start = time.time()

    for epoch in range(args.epochs):
        permutation = torch.randperm(len(x_train))
        optimizer.zero_grad()
        y_pred = torch.stack([model(x) for x in x_train[permutation]]).squeeze(-1)
        #y_pred = y_pred.to(device)
        loss = criterion(y_pred, y_train[permutation])
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        torch.save(model.state_dict(),
                   os.path.join(params_dir, f"epoch_{epoch:03d}.pt"))

        log.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    time_end = time.time()
    duration = time_end - time_start

    np.save(os.path.join(results_dir, "loss_history.npy"), np.array(loss_history))

    train_preds = predict(model, x_train)
    test_preds = predict(model, x_test)

    train_acc = compute_accuracy(y_train, train_preds)
    test_acc = compute_accuracy(y_test, test_preds)

    np.save(os.path.join(results_dir, "train_predictions.npy"), train_preds)
    np.save(os.path.join(results_dir, "test_predictions.npy"), test_preds)

    plot_predictions(
        x_train.detach().numpy(),
        train_preds,
        title="train_predictions_plot",
        outdir=results_dir
    ) 

    plot_predictions(
        x_test.detach().numpy(),
        test_preds,
        title="test_predictions_plot",
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
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "duration": duration,
        "x_train": x_train.detach().numpy().tolist(),
        "train_predictions": train_preds,
        "x_test": x_train.detach().numpy().tolist(),
        "test_predictions": test_preds,
    }

    with open(os.path.join(results_dir, f"report_data.json"), "w") as file:
        json.dump(report_data, file, indent=4)
