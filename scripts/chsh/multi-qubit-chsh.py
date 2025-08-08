import pathlib
import os
import json

os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

import numpy as np
import matplotlib.pyplot as plt
from qibo import set_backend, gates, set_transpiler, Circuit
from qibo.measurements import MeasurementResult
from qibo.transpiler import (
    NativeGates,
    Passes,
    Unroller
)
from qibo.ui import plot_circuit

glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
natives = NativeGates(0).from_gatelist(glist)
custom_passes = [Unroller(native_gates=natives)]
custom_pipeline = Passes(custom_passes)

set_backend("qibolab", platform="sinq-20")
set_transpiler(custom_pipeline)

pairs = {
    (1, 0)  : [1.70992262, -2.26150014],
    (3, 2)  : [7.87767684, -9.13055863],
    (4, 9)  : [1.70843489,  4.0251965 ],
    (6, 5)  : [1.63375093,  0.6727896 ],
    (17, 13): [1.27315056,  3.44943578],
    #(18, 14): [6.94087803,  2.48933919]
}


def bell_circuit(basis, q0, q1, angle1, angle2) -> tuple[Circuit, MeasurementResult]:
    """Create a Bell circuit with a distinct measurement basis and parametrizable gates.

    Args:
        basis (str): '00', '01, '10', '11' where a '1' marks a measurement in the X basis
            and a '0' a measurement in the Z basis.

    Returns:
        :class:`qibo.core.circuit.Circuit`

    """
    circuit = Circuit(20)
    circuit.add(gates.RY(q0, theta=angle1))
    circuit.add(gates.CNOT(q0, q1))
    circuit.add(gates.RY(q0, theta=angle2))
    for base, a in zip(basis, [q0, q1]):
        if base == "1":
            circuit.add(gates.H(a))
    measurement = circuit.add(gates.M(*(q0, q1), register_name=f"{q0}-{q1}"))
    return circuit, measurement


def set_parametrized_circuits(q0, q1, angle1, angle2) -> tuple[list[Circuit], list[MeasurementResult]]:
    """Create all Bell circuits needed to compute the CHSH inequelities.
    Returns:
        list(:class:`qibo.core.circuit.Circuit`)

    """
    chsh_circuits = []
    measurements = []
    basis = ["00", "01", "10", "11"]
    for base in basis:
        circuit, measurement = bell_circuit(base, q0, q1, angle1, angle2)
        chsh_circuits.append(circuit)
        measurements.append(measurement)
    return chsh_circuits, measurements


def compute_chsh(frequencies, nshots):
    """Computes the CHSH inequelity value given the restults of an experiment.
    Args:
        frequencies (list): list of dictionaries with the result for the measurement
                            and the number of times it repeated, for each measurement
                            basis
        nshots (int): number of samples taken in each run of the circuit.

    Returns:
        chsh (float): value of the CHSH inequality.

    """
    chsh = 0
    aux = 0
    for freq in frequencies:
        for outcome in freq:
            if aux == 1:
                chsh -= (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
            else:
                chsh += (-1) ** (int(outcome[0]) + int(outcome[1])) * freq[outcome]
        aux += 1
    chsh /= nshots
    return chsh


nshots = 1024

if __name__ == "__main__":

    circuit_dict: dict[tuple[int, int], list[Circuit]] = {}
    result_dict: dict[tuple[int, int], list[MeasurementResult]] = {}
    results: dict[str, float] = {}

    for key, params in pairs.items():
        q0, q1 = key
        angle1, angle2 = params
        circuits, measurements = set_parametrized_circuits(q0, q1, angle1, angle2)
        circuit_dict[key] = circuits
        result_dict[key] = measurements

    for base in range(4):
        c = Circuit(20)

        for circuit_array in circuit_dict.values():
            c += circuit_array[base]

        # Uncomment for circuit plotting
        """
        plot_circuit(c)
        plt.savefig(f"circuit_{base}.pdf")
        """
        c(nshots=nshots)

    for pair, measurement_array in result_dict.items():
        frequencies = []

        for measurement in measurement_array:
            frequencies.append(measurement.frequencies())

        chsh_val = compute_chsh(frequencies, nshots)
        results[str(pair)] = chsh_val

    with open("bell-parallel-results.json", "w+") as w:
        w.write(json.dumps(results))
