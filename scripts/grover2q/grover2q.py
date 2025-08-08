import argparse
from qibo import Circuit, gates, set_backend
import qibo_client
from dynaconf import Dynaconf
import json

def grover_2q(qubits, target):
    c = Circuit(20)
    c.add([gates.H(i) for i in qubits])
    for i, bit in enumerate(target):
        if int(bit)==0:
            c.add(gates.X(qubits[i]))
    c.add(gates.CZ(qubits[0], qubits[1]))
    for i, bit in enumerate(target):
        if int(bit)==0:
            c.add(gates.X(qubits[i]))
    c.add([gates.H(i) for i in qubits])
    c.add([gates.X(i) for i in qubits])
    c.add(gates.CZ(qubits[0], qubits[1]))
    c.add([gates.X(i) for i in qubits])
    c.add([gates.H(i) for i in qubits])
    c.add(gates.M(*qubits, register_name=f"m{qubits}"))
    return c

def main(qubit_pairs, device, nshots, via_client=True):
    # Load credentials from .secrets.toml
    settings = Dynaconf(
        settings_files=[".secrets.toml"], environments=True, env="default"
    )
    print("Loaded settings:", settings.as_dict())
    key = settings.key

    results = dict()
    data = dict()

    qubit_pairs = [[0, 1]]
    device = "nqch"
    nshots = 1000
    client=qibo_client.Client(token=key)

    target = '11'

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
            job = client.run_circuit(c,device=device, nshots=nshots)
            r = job.result(verbose=True)
            freq = r.frequencies()
        else:
            # set_backend("qibolab", platform=device)
            set_backend("numpy")
            r = c(nshots=nshots)
            freq = r.frequencies()
        target_freq = freq[target]
        results["success_rate"][f"{qubits}"] = target_freq/nshots
        results["plotparameters"]["frequencies"][f"{qubits}"] = r.probabilities(qubits).tolist()

        with open('../../data/grover2q/data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        with open('../../data/grover2q/results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubit_pairs",
        default=[[0, 1]],
        type=list,
        help="Target qubit pairs",
    )
    parser.add_argument(
        "--device", default="nqch", type=str, help="Device to use"
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    parser.add_argument(
        "--via_client", default=True, type=bool, help="Use qibo client or direct"
    )
    args = vars(parser.parse_args())
    main(**args)