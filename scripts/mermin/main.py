import pathlib
import os
import json
import argparse
import qibo_client
from dynaconf import Dynaconf

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

# Add scripts/ to sys.path so we import scripts/config.py
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # scripts/config.py


def create_mermin_circuit(qubits):
    c = Circuit(len(qubits))
    c.add(gates.H(qubits[0]))
    c.add([gates.CNOT(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)])
    c.add(gates.RZ(qubits[0], 0))
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


def main(nqubits, qubit_list, device, nshots, via_client):
    if via_client or device == "nqch":
        # Load credentials from .secrets.toml
        settings = Dynaconf(
            settings_files=[".secrets.toml"], environments=True, env="default"
        )
        print("Loaded settings:", settings.as_dict())
        key = settings.key
        client = qibo_client.Client(token=key)

    results = dict()
    data = dict()

    results["x"] = {}
    results["y"] = {}
    data["qubit_list"] = qubit_list
    data["nqubits"] = nqubits
    data["nshots"] = nshots
    data["device"] = device

    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)

    set_backend("numpy")  # , platform="sinq-20") # , platform=device)
    set_transpiler(custom_pipeline)

    poly = get_mermin_polynomial(nqubits)
    coeff = get_mermin_coefficients(poly)
    basis = get_readout_basis(poly)

    for qubits in qubit_list:
        circuits = create_mermin_circuits(qubits, basis)
        theta_array = np.linspace(0, 2 * np.pi, 50)
        result = np.zeros(len(theta_array))
        for idx, theta in enumerate(theta_array):
            frequencies = []
            for circ in circuits:
                circ.set_parameters([theta])
                if via_client:
                    # use the correct circuit variable
                    job = client.run_circuit(circ, device=device, nshots=nshots)
                    freq = job.result(verbose=True).frequencies()
                else:
                    freq = circ(nshots=nshots).frequencies()
                frequencies.append(freq)
            result[idx] = compute_mermin(frequencies, coeff)

        results["x"][f"{qubits}"] = theta_array.tolist()
        results["y"][f"{qubits}"] = result.tolist()

        # Write to data/<scriptname>/<device>/results.json
        out_dir = config.output_dir_for(__file__) / device
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        with open(out_dir / f"data_mermin_{nqubits}q.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits",
        default=3,
        type=int,
        help="Total number of qubits",
    )
    parser.add_argument(
        "--qubit_list",
        default=[[0, 1, 2]],
        type=list,
        help="Target qubits list",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use",
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
