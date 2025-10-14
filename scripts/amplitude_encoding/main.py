import numpy as np
import qibo
import os
import json
import sys
from pathlib import Path as _P
import time


sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def bitcount(arr):
    """Count set bits in each integer of a NumPy array."""
    return np.frompyfunc(lambda x: bin(x).count("1"), 1, 1)(arr).astype(np.int64)


def walsh_gray_matrix(k: int, dtype=np.float64) -> np.ndarray:
    """
    Return the 2^k x 2^k matrix M with entries
    M[i, j] = 2^{-k} * (-1)^{b_j · g_i},
    where b_j is the binary code of j, g_i is the Gray code of i,
    and the dot-product is mod 2.
    """
    n = 1 << k  # 2^k
    I = np.arange(n, dtype=np.uint64)  # 0..2^k-1
    J = np.arange(n, dtype=np.uint64)

    GI = I ^ (I >> 1)  # Gray codes for i

    # Parity(b_j · g_i) = bitcount(g_i & j) mod 2
    # Broadcast to an (n x n) array without Python loops
    # bitcount(g_i & j), mod 2
    parity = bitcount(GI[:, None] & J[None, :]) & 1

    M = ((-1) ** parity).astype(dtype) * (2.0**-k)
    return M


def F_k(circuit, k, qubits_list, theta):
    """
    Return the uniformly controlled rotation F_k^(k+1) circuit with angles
    gate defined by a sequence of all possible controlled rotations Ra (γi ),
    on the target qubit k+1 controlled by the k qubits
    see https://arxiv.org/abs/quant-ph/0407010
    """
    circuit.add(qibo.gates.RY(qubits_list[k], theta[0]))

    if k == 0:
        return circuit
    else:
        seq = []
        for n in range(1, k + 1):  # build up sequence for CNOT application
            seq = seq + seq[:-1] + [n]

        # Build circuit
        for i, s in enumerate(seq):
            circuit.add(qibo.gates.CNOT(qubits_list[k - s], qubits_list[k]))
            circuit.add(qibo.gates.RY(qubits_list[k], theta[i + 1]))

        for i, s in enumerate(seq):
            circuit.add(qibo.gates.CNOT(qubits_list[k - s], qubits_list[k]))

            if i == len(seq) - 1:
                break
            else:
                circuit.add(qibo.gates.RY(qubits_list[k], theta[i + 1 + len(seq)]))

        return circuit


def amplitude_enc(vector, edges, nshosts):
    """
    Args: - a vector of numerical values (real for the moments)
          - list of qubits where to apply the amp_encoding
          - nshots
    Return: circuit.result. Amplitude must be the normalized elements in
            input vector
    """
    
    qubits_list = sorted(set(sum(edges, [])))

    n = len(qubits_list)  # num qubits used

    if len(vector) != 2**n:
        raise ValueError("Input vector must be 2^num_qubits or use padding")

    # Normalization of elements in vector
    vec_norm = vector / np.linalg.norm(vector)

    total_qubits = int(np.max(qubits_list) + 1)

    circuit = qibo.Circuit(nqubits=total_qubits)

    # Angle calculation and building up the circuit
    # for Amplitude Encoding
    # see https://arxiv.org/abs/quant-ph/0407010
    for k in range(n):
        M = walsh_gray_matrix(k)

        gamma = np.zeros(2**k)
        theta = np.zeros(2**k)
        for i in range(2**k):

            numerator = np.array(
                [
                    vec_norm[(2 * i + 1) * 2 ** (n - k - 1) + l]
                    for l in range(2 ** (n - k - 1))
                ]
            )

            denominator = np.array(
                [vec_norm[i * 2 ** (n - k) + l] for l in range(2 ** (n - k))]
            )

            gamma[i] = 2 * np.asin(
                np.sqrt(
                    np.sum(np.abs(numerator) ** 2) / np.sum(np.abs(denominator) ** 2)
                )
            )

        theta = M @ gamma

        F_k(circuit, k, qubits_list, theta)

    for q in qubits_list:
        circuit.add(qibo.gates.M(q))

    start = time.perf_counter()
    result = circuit(nshots=nshosts)
    end = time.perf_counter()

    return result, circuit.depth, len(circuit.queue), end - start


def main(vector, edges, device, nshots):
    # Remove all qibo_client usage and via_client logic
    # Set backend as in template/main.py, GHZ/main.py, mermin/main.py
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)

    results = dict()
    data = dict()

    results["description"] = {}
    results["input_vector"] = vector
    results["circuit_depth"] = {}
    results["gates_count"] = {}
    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubits_list"] = edges
    data["nshots"] = nshots
    data["device"] = device

    result, depth, num_gates, duration = amplitude_enc(vector, edges, nshots)

    qubits_list = sorted(set(sum(edges, [])))

    n_qubits = len(qubits_list)
    success_keys = ["0" * n_qubits, "1" * n_qubits]
    total_success = sum(result.frequencies().get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
    freq_dict = {
        bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings
    }

    results = {
        "description": f"Encoding of a vector of numerical data into the amplitudes of a quantum state. The input vector is {vector}. The number of gates is {num_gates}, the depth of the circuit is {depth} and the runtime execution is {duration:.3f}ms",
        "input_vector": vector,
        "circuit_depth": depth,
        "gates_count": num_gates,
        "runtime": f"{duration:.2f} seconds.",
        "success_rate": success_rate,
        "qubits_used": edges,
        "plotparameters": {"frequencies": freq_dict},
    }

    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f"File saved on {output_path}")


import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_vector",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        type=float,
        nargs="+",
        help="Input of numerical vector to encode into the amplitude",
    )
    parser.add_argument(
        "--edges",
        default="[[9,8],[8,13]]",
        type=list,
        help="Target edges list as string representation",
    )
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
    args = parser.parse_args()
    # Parse the qubit list string into actual list of integers
    try:
        edges = ast.literal_eval(args.edges)
        # Ensure all elements are integers
        edges = [[int(q) for q in edge] for edge in edges]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.edges}")
        sys.exit(1)
    main(args.input_vector, args.edges, args.device, args.nshots)