import pathlib
import os
import json

# os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

import numpy as np
import matplotlib.pyplot as plt
from qibo import Circuit, gates, set_transpiler, set_backend
from qibo.transpiler import NativeGates, Passes, Unroller

from utils import (
    get_mermin_coefficients,
    get_mermin_polynomial,
    get_readout_basis,
    compute_mermin,
)

glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
natives = NativeGates(0).from_gatelist(glist)
custom_passes = [Unroller(native_gates=natives)]
custom_pipeline = Passes(custom_passes)

set_backend("numpy")  # , platform="sinq-20")
set_transpiler(custom_pipeline)


def create_mermin_circuit(qubits):
    c = Circuit(5)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CNOT(1, 2))
    c.add(gates.CNOT(2, 3))
    c.add(gates.CNOT(3, 4))
    c.add(gates.RZ(2, 0))
    return c


def create_mermin_circuits(qubits: list[int], readout_basis: list[str]):
    c = create_mermin_circuit(qubits)
    circuits = [c.copy(deep=True) for _ in readout_basis]

    for circuit, basis in zip(circuits, readout_basis):
        for q, base in zip(qubits, basis):
            if base == "Y":
                circuit.add(gates.SDG(q))
            circuit.add(gates.H(q))
            circuit.add(gates.M(q))

    return circuits


poly = get_mermin_polynomial(5)
coeff = get_mermin_coefficients(poly)
basis = get_readout_basis(poly)

circuits = create_mermin_circuits([0, 1, 2, 3, 4], basis)
theta_array = np.linspace(0, 2 * np.pi, 50)
results = np.zeros(len(theta_array))
nshots = 1000

for idx, theta in enumerate(theta_array):
    frequencies = []
    for circ in circuits:
        circ.set_parameters([theta])
        frequencies.append(circ(nshots=nshots).frequencies())
    results[idx] = compute_mermin(frequencies, coeff)

with open("data/mermin_5q.json", "w+") as w:
    w.write(json.dumps({"x": theta_array.tolist(), "y": results.tolist()}))
