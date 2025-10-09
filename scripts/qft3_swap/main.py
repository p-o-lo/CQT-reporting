import numpy as np
import qibo
import os
import json
import sys
import time
from pathlib import Path as _P


sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def QFT_swap(qubits_list):
    '''
    Quantum Fourier Transform circuit from scratch (ONLY THREE QUBITS)
    
    Args:   - list of selected qubits set
            - device (str): device where to execute the circuit (local numpy,
              qibo_client sinq20)
            
    Returns: qibo circuit
    '''
    
    total_qubits = int(np.max(qubits_list) + 1)

    circuit = qibo.Circuit(total_qubits)                # Circuit initialization

    # Add Hadamard at the beginning
    for q in qubits_list:
        circuit.add(qibo.gates.H(q))
    
    # QFT from scratch for three qubits with swap for 
    # connection q0--q1--q2
    circuit.add(qibo.gates.H(qubits_list[0]))
    theta1 = np.pi / 2
    circuit.add(qibo.gates.CU1(qubits_list[1], qubits_list[0], theta1))
    circuit.add(qibo.gates.SWAP(qubits_list[1], qubits_list[2]))
    theta2 = np.pi / 4
    circuit.add(qibo.gates.CU1(qubits_list[1], qubits_list[0], theta2))
    circuit.add(qibo.gates.H(qubits_list[2]))
    circuit.add(qibo.gates.CU1(qubits_list[1], qubits_list[2], theta1))
    circuit.add(qibo.gates.H(qubits_list[1]))
    
    # Add measurement
    for q in qubits_list:
        circuit.add(qibo.gates.M(q))

    return circuit


def main(device, nshots):
    # Remove all qibo_client usage and via_client logic
    # Set backend as in template/main.py, GHZ/main.py, mermin/main.py
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)

    # Here the list of the best three qubits
    qubits_lists = [
        [14, 17, 18],
    ]

    num_qubits = len(qubits_lists[0])

    frequencies = dict()
    fidelities = []
    times = []
    all_bitstrings = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]

    for qubits_list in qubits_lists:
        print(f"Trying qubits: {qubits_list}")

        circuit = QFT_swap(qubits_list)

        start = time.perf_counter()
        result = circuit(nshots=nshots)
        end = time.perf_counter()

        circuit_state = (
            np.array([result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings])
            / nshots
        )

        key = format(qubits_list)
        frequencies[key] = circuit_state
        fidelities.append(circuit_state[0])
        times.append(end - start)

    num_gates = len(circuit.queue)
    depth = circuit.depth

    results = dict()
    data = dict()

    frequencies = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in frequencies.items()
    }


    results["description"] = {}
    results["circuit_depth"] = {}
    results["gates_count"] = {}
    results["duration"] = {}
    results["frequencies"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["qubits_lists"] = {}
    results["plotparameters"]["fidelities"] = {}
    data["nshots"] = nshots
    data["device"] = device

    results = {
        "description": f"Implementation of the Quantum Fourier Transform on different subsets of three qubits. The number of gates is {num_gates}, the depth of the circuit is {depth}",
        "circuit_depths": depth,
        "gates_counts": num_gates,
        "runtime": f"{np.average(times):.2f} seconds.",
        "frequencies": frequencies,
        "qubits_used": qubits_lists,
        "plotparameters": {"qubits_lists": qubits_lists, "fidelities": fidelities},
    }
    

    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f"File saved on {output_path}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="numpy",
        type=str,
        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = vars(parser.parse_args())
    main(args["device"], args["nshots"])