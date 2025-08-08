import argparse
import numpy as np
import pathlib
import os

os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

from functions import compute_chsh, cost_function, set_parametrized_circuits
from qibo import set_backend, gates, set_transpiler
from qibo.transpiler import (
    NativeGates,
    Passes,
    Unroller
)
from qibo.optimizers import optimize

glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
natives = NativeGates(0).from_gatelist(glist)
custom_passes = [Unroller(native_gates=natives)]
custom_pipeline = Passes(custom_passes)

set_backend("qibolab", platform="sinq-20")
set_transpiler(custom_pipeline)


nshots = 1024

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("q0", type=int)
    parser.add_argument("q1", type=int)

    args = parser.parse_args()
    q0 = args.q0
    q1 = args.q1

    initial_parameters = np.random.uniform(0, 2 * np.pi, 2)
    circuits = set_parametrized_circuits(wire_names=[q0, q1])
    best, params, _ = optimize(
        cost_function, initial_parameters, args=(circuits, nshots)
    )
    outstr = ""
    outstr += (f"Cost: {best}\n")
    outstr += (f"Parameters: {params}\n")
    outstr += (f"Angles for the RY gates: {(params*180/np.pi)%360} in degrees.\n")
    frequencies = []
    for circuit in circuits:
        circuit.set_parameters(params)
        frequencies.append(circuit(nshots=nshots).frequencies())
    chsh = compute_chsh(frequencies, nshots)
    outstr += (f"CHSH inequality value: {chsh}\n")
    outstr += (f"Target: {np.sqrt(2)*2}\n")
    outstr += (f"Relative distance: {100*np.abs(np.abs(chsh)-np.sqrt(2)*2)/np.sqrt(2)*2}%\n")
   
    with open(f"{q0}-{q1}.txt", "w+") as f:
        f.write(outstr)
