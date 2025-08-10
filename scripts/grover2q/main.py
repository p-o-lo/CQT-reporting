import argparse
from qibo import Circuit, gates, set_backend
import qibo_client
from dynaconf import Dynaconf
import json
from pathlib import Path


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


def main(qubit_pairs, device, nshots, via_client=True):
    if via_client:
        # Load credentials from .secrets.toml
        settings = Dynaconf(
            settings_files=[".secrets.toml"], environments=True, env="default"
        )
        print("Loaded settings:", settings.as_dict())
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
        if via_client:
            job = client.run_circuit(c, device=device, nshots=nshots)
            r = job.result(verbose=True)
            freq = r.frequencies()
        else:
            # Support local simulation via numpy backend
            if device == "numpy":
                set_backend("numpy")
            # else:
            #     set_backend("qibolab", platform=device)
            r = c(nshots=nshots)
            freq = r.frequencies()

        target_freq = freq.get(target, 0)
        results["success_rate"][f"{qubits}"] = target_freq / nshots

        # Make probabilities a dict keyed by all possible bitstrings
        num_bits = len(qubits)
        all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
        prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
        results["plotparameters"]["frequencies"][f"{qubits}"] = prob_dict

    # Robust file writing to project-root/data/grover2q
    repo_root = Path(__file__).resolve().parents[2]  # .../quantum-benchmark-reporter
    out_dir = repo_root / "data" / "grover2q"
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
        default="nqch",
        type=str,
        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    parser.add_argument(
        "--via_client", default=False, type=bool, help="Use qibo client or direct"
    )
    args = vars(parser.parse_args())
    main(**args)
