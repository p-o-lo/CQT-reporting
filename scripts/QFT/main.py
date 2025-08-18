import numpy as np
import qibo
import qibo_client
import os
import json
from dynaconf import Dynaconf

def QFT(qubits_list, device, nshosts, via_client, client):                      
    
    result = None
    n_qubits = len(qubits_list)
        
    # QFT on local sim
    if not via_client:
        total_qubits = int(np.max(qubits_list) + 1)
    # QFT on remote QPU
    if via_client:
        total_qubits = 20
    
    # QFT
    circuit = qibo.Circuit(total_qubits)
    for k in range(n_qubits):
        qk = qubits_list[k]
        circuit.add(qibo.gates.H(qk))
        for j in range(k + 1, n_qubits):
            qj = qubits_list[j]
            theta = np.pi / (2 ** (j - k))
            circuit.add(qibo.gates.CU1(qk, qj, theta))

    # Add measurement
    for q in range(total_qubits):
        circuit.add(qibo.gates.M(q))

    # Run on local sim
    if not via_client:
        result = circuit(nshots=nshosts)
    
    # Run on remote QPU
    if via_client:
        job = client.run_circuit(circuit, device=device, project="personal", nshots=nshosts)
        result = job.result(verbose=True)
    
    return result


def main(qubits_list, device, nshots, via_client):
    client = None
    if via_client:
        # Load credentials from .secrets.toml
        settings = Dynaconf(
            settings_files=[".secrets.toml"], environments=True, env="default"
        )
        print("Loaded key")
        key = settings.key
        client = qibo_client.Client(token=key)
        
    results = dict()
    data = dict()

    results["success_rate"] = {}
    results["plotparameters"] = {}
    results["plotparameters"]["frequencies"] = {}
    data["qubits_list"] = qubits_list
    data["nshots"] = nshots
    data["device"] = device

    # On local simulator
    if not via_client:
        total_qubits = int(np.max(qubits_list) + 1)
        qibo.set_backend(backend=device)
    # on QPU
    if via_client:
        total_qubits = 20

    result = QFT(qubits_list, device, nshots, via_client, client)    

    success_keys = ["0" * total_qubits, "1" * total_qubits ]
    total_success = sum(result.frequencies().get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    all_bitstrings = [format(i, f"0{total_qubits}b") for i in range(2**total_qubits)]
    freq_dict = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

    results = {
            "success_rate": success_rate,
            "plotparameters": {"frequencies": freq_dict},
        }

    output_dir = f"../../data/QFT/{device}"
    os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir,
    # f"QFT_on_{qubits_list}_{nshots}shots_device_{device}.json")
    output_path = os.path.join(output_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f'File saved on {output_path}')
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        default=[0, 1, 2, 3, 4],
        type=int,
        nargs='+',
        help="List of qubits exploited in the device",
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
    parser.add_argument(
        "--via_client", default=False, type=bool, help="Use qibo client or direct"
    )
    args = vars(parser.parse_args())
    main(**args)