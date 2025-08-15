import argparse
from qibo import Circuit, gates, set_backend
import qibo_client
from dynaconf import Dynaconf
import json
from pathlib import Path
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def grover_2q(qubits, target):
    c = Circuit(20)
    c.add([gates.H(i) for i in qubits])
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add(gates.CZ(qubits[0], qubits[1]))
    for i, bit in enumerate(target):
        if int(bit) == 0:
            c.add(gates.X(qubits[i]))
    c.add([gates.H(i) for i in qubits])
    c.add([gates.X(i) for i in qubits])
    c.add(gates.CZ(qubits[0], qubits[1]))
    c.add([gates.X(i) for i in qubits])
    c.add([gates.H(i) for i in qubits])
    c.add(gates.M(*qubits, register_name=f"m{qubits}"))
    return c


def main(qubit_pairs, device, nshots):
    if device == "nqch":
        settings = Dynaconf(
            settings_files=[".secrets.toml"], environments=True, env="default"
        )
        key = settings.key
        client = qibo_client.Client(token=key)

    results = dict()
    data = dict()

    target = "11"

    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubit_pairs"] = qubit_pairs
    data["nshots"] = nshots
    data["device"] = device
    data["target"] = target

    for qubits in qubit_pairs:
        c = grover_2q(qubits, target)
        if device == "nqch":
            job = client.run_circuit(c, device=device, nshots=nshots)
            r = job.result(verbose=True)
            freq = r.frequencies()
        elif device == "numpy":
            # Support local simulation via numpy backend
            set_backend("numpy")
            r = c(nshots=nshots)
            freq = r.frequencies()

        target_freq = freq.get(target, 0)
        results["success_rate"][f"{qubits}"] = target_freq / nshots

        # Make probabilities a dict keyed by all possible bitstrings
        num_bits = len(qubits)
        all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
        prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
        results["plotparameters"]["frequencies"][f"{qubits}"] = prob_dict

    # Write to data/<scriptname>/<device>/results.json
    out_dir = config.output_dir_for(__file__) / device
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (out_dir / "data.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubit_pairs",
        default=[[0, 1]],
        type=list,
        help="Target qubit pairs",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use (numpy, nqch-sim, or sinq20)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = vars(parser.parse_args())
    main(**args)
