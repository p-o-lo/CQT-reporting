import numpy as np
import qibo
import qibo_client
import os
import json
from dynaconf import Dynaconf

def QFT(qubits_list, device, nshosts, via_client, client):                      
    
    result = None
    n_qubits = len(qubits_list)
    total_qubits = int(np.max(qubits_list) + 1)
    
    # QFT on local sim
    if not via_client:
        circuit = qibo.Circuit(total_qubits)
    # QFT remotely on QPU
    if via_client:
        circuit = qibo.Circuit(20)
    
    ## Circuit Definition
    # Add Hadamard at the beginning
    for q in qubits_list:
        circuit.add(qibo.gates.H(q)) 
    # QFT
    qft_circuit = qibo.models.QFT(n_qubits, with_swaps=False)
    circuit.add(qft_circuit.on_qubits(*qubits_list))
    
    # Add measurement
    for q in qubits_list:
        circuit.add(qibo.gates.M(q))
            
    # QFT on remote QPU
    # if via_client:
    #     total_qubits = 20

    # # QFT
    # circuit = qibo.Circuit(total_qubits)
    # for k in range(n_qubits):
    #     qk = qubits_list[k]
    #     # circuit.add(qibo.gates.H(qk))
    #     for j in range(k + 1, n_qubits):
    #         qj = qubits_list[j]
    #         theta = np.pi / (2 ** (j - k))
    #         circuit.add(qibo.gates.CU1(qk, qj, theta))

    # # Run on local sim
    # if not via_client:
    #     result = circuit(nshots=nshosts)
    
    # Run locally
    if not via_client:
        result = circuit(nshosts=nshosts)
    
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

    # total_qubits = int(np.max(qubits_list) + 1)

    # On local simulator
    if not via_client:    
        qibo.set_backend(backend=device)
    # on QPU
    # if via_client:
    #     total_qubits = 20

    result = QFT(qubits_list, device, nshots, via_client, client)    

    n_qubits = len(qubits_list)

    success_keys = ["0" * n_qubits, "1" * n_qubits ]
    total_success = sum(result.frequencies().get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]
    freq_dict = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

    results = {
            "success_rate": success_rate,
            "plotparameters": {"frequencies": freq_dict},
        }

    output_dir = f"../../data/QFT/{device}"
    os.makedirs(output_dir, exist_ok=True)
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