import numpy as np
import qibo
import os
import json

qibo.set_backend(backend="numpy")

n_qubits = 5
n_shots = 1024

circuit = qibo.models.QFT(n_qubits)
for q in range(n_qubits):
    circuit.add(qibo.gates.M(q))
    
result = circuit(nshots=n_shots)

success_keys = ["0" * n_qubits, "1" * n_qubits]
total_success = sum(result.frequencies().get(k, 0) for k in success_keys)
success_rate = total_success / n_shots if n_shots else 0.0

all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
freq_dict = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

results = {
        "success_rate": success_rate,
        "plotparameters": {"frequencies": freq_dict},
    }

output_dir = os.path.join("data", "QFT_sim")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"QFT_{n_qubits}q_{n_shots}shots.json")

with open(output_path, "w") as f:
    json.dump(results, f)