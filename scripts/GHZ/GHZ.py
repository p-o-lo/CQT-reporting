import json
import numpy as np
from qibo import Circuit, gates, set_backend
import os

set_backend("numpy")


def create_ghz_circuit(nqubits: int):
    c = Circuit(nqubits)
    c.add(gates.H(0))
    for i in range(nqubits - 1):
        c.add(gates.CNOT(i, i + 1))
    for i in range(nqubits):
        c.add(gates.M(i))
    return c


nqubits = 5
nshots = 1000

circuit = create_ghz_circuit(nqubits)
result = circuit(nshots=nshots)
frequencies = result.frequencies()


def prepare_ghz_results(frequencies, nshots, nqubits, circuit):
    # Calculate success rate for GHZ state (all 0s or all 1s)
    success_keys = ["0" * nqubits, "1" * nqubits]
    total_success = sum(frequencies.get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    # Prepare output structure
    all_bitstrings = [format(i, f"0{nqubits}b") for i in range(2**nqubits)]
    freq_dict = {bitstr: frequencies.get(bitstr, 0) for bitstr in all_bitstrings}
    results = {
        "success_rate": success_rate,
        "plotparameters": {"frequencies": freq_dict},
    }
    return results


results = prepare_ghz_results(frequencies, nshots, nqubits, circuit)

output_dir = os.path.join("data", "ghz")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ghz_5q_samples.json")

with open(output_path, "w") as f:
    json.dump(results, f)
