import json
import numpy as np
from qibo import Circuit, gates, set_backend
import os
import argparse
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def create_ghz_circuit(nqubits: int):
    c = Circuit(nqubits)
    c.add(gates.H(0))
    for i in range(nqubits - 1):
        c.add(gates.CNOT(i, i + 1))
    for i in range(nqubits):
        c.add(gates.M(i))
    return c


def prepare_ghz_results(frequencies, nshots, nqubits, circuit):
    # Calculate success rate for GHZ state (all 0s or all 1s)
    success_keys = ["0" * nqubits, "1" * nqubits]
    total_success = sum(frequencies.get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    # Prepare output structure
    all_bitstrings = [format(i, f"0{nqubits}b") for i in range(2**nqubits)]
    freq_dict = {bitstr: frequencies.get(bitstr, 0) for bitstr in all_bitstrings}
    return {"success_rate": success_rate, "plotparameters": {"frequencies": freq_dict}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=5)
    parser.add_argument("--nshots", type=int, default=1000)
    parser.add_argument(
        "--device", choices=["numpy", "nqch-sim", "sinq20"], default="numpy"
    )
    args = parser.parse_args()

    if args.device == "numpy":
        set_backend("numpy")
    # else: keep default behavior; simulation only

    circuit = create_ghz_circuit(args.nqubits)
    result = circuit(nshots=args.nshots)
    frequencies = result.frequencies()
    results = prepare_ghz_results(frequencies, args.nshots, args.nqubits, circuit)

    out_dir = config.output_dir_for(__file__) / args.device
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
