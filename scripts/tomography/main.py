import os
import pathlib
import argparse
import json
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # scripts/config.py

# os.environ["QIBOLAB_PLATFORMS"] = (pathlib.Path(__file__).parent / "qibolab_platforms_nqch").as_posix()

from qibo import Circuit, gates, set_backend
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
import numpy as np

targets = (13, 8)
root_path = "tomography_now_report"
platform = "sinq-20"

with Executor.open(
    "myexec",
    path=root_path,
    platform=platform,
    targets=[targets],
    update=True,
    force=True,
) as e:
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    output = e.two_qubit_state_tomography(circuit=circuit, targets=[targets])
    report(e.path, e.history)
