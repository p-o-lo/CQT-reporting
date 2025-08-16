from inspect import signature
from itertools import product
from typing import List, Optional, Tuple, Union
from functools import cache
import timeit

import numpy as np
from sympy import S

from qibo import Circuit, gates, symbols, construct_backend
from qibo.backends import _check_backend, NumpyBackend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller

from scipy.linalg import norm
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

from qibo.noise import NoiseModel, DepolarizingError

noise_model = NoiseModel()
noise_model.add(DepolarizingError(0.3))
import os
import argparse
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


SUPPORTED_NQUBITS = [1, 2]
"""Supported nqubits for GST."""


ANGLES = ["theta", "phi", "lam", "unitary"]
"""Angle names for parametrized gates."""


# @cache
def _check_nqubits(nqubits):
    if nqubits not in SUPPORTED_NQUBITS:
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )


# @cache
def _gates(nqubits) -> List:
    """Gates implementing all the GST state preparations.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        List(:class:`qibo.gates.Gate`): gates used to prepare the possible states.
    """
    ## For native gates
    # ANGLES = ["theta", "phi", "lam"]
    # gates_list = [(gates.I,), (gates.GPI2, gates.GPI2, gates.Z,), (gates.Z, gates.GPI2,), (gates.Z, gates.GPI2, gates.RZ,)]
    # angles = [(np.nan,) , (np.pi/2, np.pi/2, np.nan,)       , (np.nan, np.pi/2,)    , (np.nan, np.pi/2, np.pi/2, )]
    # k_to_gates = list(product(gates_list, repeat=len(wire_names)))[k]
    # k_to_angles = list(product(angles, repeat=len(wire_names)))[k]
    return list(
        product(
            [(gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S)], repeat=nqubits
        )
    )


# @cache
def _measurements(nqubits: int) -> List:
    """Measurement gates implementing all the GST measurement bases.

    Args:
        nqubits (int): Number of qubits for the circuit.
    Returns:
        List(:class:`qibo.gates.Gate`): gates implementing the possible measurement bases.
    """
    ## For native gates
    # ANGLES = ["theta", "phi", "lam"]
    # meas_list = [(gates.I,), (gates.Z, gates.GPI2,), (gates.RZ, gates.Z, gates.GPI2,), (gates.I,)]
    # angles = [(np.nan,) , (np.nan, np.pi/2,)    , (-np.pi/2, np.nan, np.pi/2,)    , (np.nan,)]
    # j_to_measurements = list(product(meas_list, repeat=len(wire_names)))[j]
    # j_to_angles = list(product(angles, repeat=len(wire_names)))[j]
    return list(product([gates.Z, gates.X, gates.Y, gates.Z], repeat=nqubits))


# @cache
def _observables(nqubits: int) -> List:
    """All the observables measured in the GST protocol.

    Args:
        nqubits (int): number of qubits for the circuit.

    Returns:
        List[:class:`qibo.symbols.Symbol`]: all possible observables to be measured.
    """

    return list(product([symbols.I, symbols.Z, symbols.Z, symbols.Z], repeat=nqubits))


# @cache
def _get_observable(j: int, nqubits: int):
    """Returns the :math:`j`-th observable. The :math:`j`-th observable is expressed as a base-4 indexing and is given by

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis (in base-4)
        nqubits (int): number of qubits.

    Returns:
        List[:class:`qibo.hamiltonians.SymbolicHamiltonian`]: observables represented by symbolic Hamiltonians.
    """

    if j == 0:
        _check_nqubits(nqubits)
    observables = _observables(nqubits)[j]
    observable = S(1)
    for q, obs in enumerate(observables):
        if obs is not symbols.I:
            observable *= obs(q)
    return SymbolicHamiltonian(observable, nqubits=nqubits)


# @cache
def _prepare_state(k: int, nqubits: int):
    """Prepares the :math:`k`-th state for an :math:`n`-qubits (`nqubits`) circuit.
    Using base-4 indexing for :math:`k`,

    .. math::
        k \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ 0\\rangle\\langle0|, |1\\rangle\\langle1|,
        |+\\rangle\\langle +|, |y+\\rangle\\langle y+|\\}^{\\otimes n}.

    Args:
        k (int): index of the state to be prepared.
        nqubits (int): Number of qubits.

    Returns:
        list(:class:`qibo.gates.Gate`): gates that prepare the :math:`k`-th state.
    """

    _check_nqubits(nqubits)
    gates = _gates(nqubits)[k]
    return [gate(q) for q in range(len(gates)) for gate in gates[q]]


# @cache
def _measurement_basis(j: int, nqubits: int):
    """Constructs the :math:`j`-th measurement basis element for an :math:`n`-qubits (`nqubits`) circuit.
    Base-4 indexing is used for the :math:`j`-th measurement basis and is given by

    .. math::
        j \\in \\{0, 1, 2, 3\\}^{\\otimes n} \\equiv \\{ I, X, Y, Z\\}^{\\otimes n}.

    Args:
        j (int): index of the measurement basis element.
        nqubits (int): number of qubits.

    Returns:
        List[:class:`qibo.gates.Gate`]: gates forming the :math:`j`-th element
            of the Pauli measurement basis.
    """

    _check_nqubits(nqubits)
    measurements = _measurements(nqubits)[j]
    return [gates.M(q, basis=measurements[q]) for q in range(len(measurements))]


def _extract_nqubits(
    gate: Union[gates.abstract.Gate, Tuple[gates.abstract.Gate, List[float]]],
):
    """A function to extract the number of qubits the gate acts on.
    Args:
        gate (:class:`qibo.gates.abstract.Gate` or tuple): Either a gate or a tuple consisting of a gate and a list of its parameters.
            Examples of a valid input:
            - ``gate = gates.Z`` for a non-parametrized gate.
            - ``gate = (gates.RX, [np.pi/3])`` or ``gate = (gates.PRX, [np.pi/2, np.pi/3])`` for a parametrized gate.
            - ``gate = (gates.Unitary, [np.array([[1, 0], [0, 1]])])`` for an arbitrary unitary operator.
    Returns:
        nqubits (int): Number of qubits that the gate acts on.
    """

    nqubits = None
    params = None
    if isinstance(gate, tuple):
        gate, params = gate
    init_args = signature(gate).parameters
    if "unitary" in init_args:
        nqubits = int(np.log2(np.shape(params[0])[0]))
    else:
        if "q" in init_args:
            nqubits = 1
        elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
            nqubits = 2
        else:
            raise_error(
                RuntimeError,
                f"Gate {gate} is not supported for `GST`, only 1- and 2-qubit gates are supported.",
            )
    return nqubits


def _get_nqubits_and_angles(
    gate: Union[gates.abstract.Gate, Tuple[gates.abstract.Gate, List[float]]],
):
    """A function to extract information about a `qibo.gates.Gate`.

    Args:
        gate (:class:`qibo.gates.abstract.Gate` or tuple): Either a gate or a tuple consisting of a gate and a list of its parameters.
            Examples of a valid input:
            - ``gate = gates.Z`` for a non-parametrized gate.
            - ``gate = (gates.RX, [np.pi/3])`` or ``gate = (gates.PRX, [np.pi/2, np.pi/3])`` for a parametrized gate.
            - ``gate = (gates.Unitary, [np.array([[1, 0], [0, 1]])])`` for an arbitrary unitary operator.
    Returns:
        gate (:class:`qibo.gates.Gate`): Gate class.
        nqubits (int): Number of qubits that the gate acts on.
        angle_names (list[str]): If gate is a parametrized gate, ``angle_names`` contains a list containing the angle names of the
            parametrized gate. Else, ``None``.
        angle_values (dict[str, float]): If gate is a parametrized gate, ``angle_values`` is a dictionary containing the angle names
            of the parametrized gate and the respective angles. Else, an empty dictionary is returned.
        params (list[float]): Stores all the parameters of the gate in a list.
    """

    nqubits = None
    original_gate = gate
    if isinstance(gate, tuple):
        angles = ANGLES
        gate, params = gate
        if not (isinstance(params, list) or isinstance(params, tuple)):
            if isinstance(params, np.ndarray):
                params = [params]
    else:
        angles = None
        params = None
    init_args = signature(gate).parameters
    nqubits = _extract_nqubits(original_gate)

    if angles:
        angle_names = [arg for arg in init_args if arg in angles]
        angle_values = dict(zip(angle_names, params))
    else:
        angle_names = None
        angle_values = {}

    return gate, nqubits, angle_names, angle_values, params


def _extract_gate(
    gate: Union[gates.abstract.Gate, Tuple[gates.abstract.Gate, List[float]]],
    idx: Optional[Union[int, Tuple[int, ...]]] = None,
):
    """Receives a gate class / tuple of gate class and parameters and extracts an instance of a
        `qibo.gates.Gate` that can be applied directly to the circuit while also returning the number of
        qubits that the gate acts on.

    Args:
        gate (type or tuple): A gate class or a tuple consisting of the class and a list of its parameters.
            Examples of a valid input:
            - `gate = gates.Z` for a non-parametrized gate.
            - `gate = (gates.RX, [np.pi/3])` or `gate = (gates.PRX, [np.pi/2, np.pi/3])` for a parametrized gate.
            - `gate = (gates.Unitary, [np.array([[1, 0], [0, 1]])])` for an arbitrary unitary operator.
        idx (int or tuple, optional): Specifies the qubit index (or indices) the gate should be applied to.
            Defaults to None, in which case qubit 0 (or qubits 0 and 1 for two-qubit gates) will be used by default.

    Returns:
        gate (:class:`qibo.gates.Gate`): An instance of the gate that can be applied directly to the circuit.
        nqubits (int): The number of qubits that the gate acts on.
    """

    original_gate_input = gate

    gate, nqubits, angle_names, angle_values, params = _get_nqubits_and_angles(
        original_gate_input
    )

    # Perform some checks
    if isinstance(original_gate_input, tuple):
        if "unitary" in angle_names:
            # Check that unitary gate does not receive a non-unitary matrix.
            g = gate(angle_values["unitary"], *range(nqubits), check_unitary=True)
            if not g.unitary:
                raise_error(ValueError, "Unitary gate received non-unitary matrix.")

    # Construct gate instance
    idx = (
        range(nqubits)
        if idx is None
        else ((idx,) if isinstance(idx, int) else tuple(idx))
    )
    if "unitary" in angle_values:
        gate = gate(angle_values["unitary"], *idx)
    else:
        gate = gate(*idx, **angle_values)

    return gate, nqubits


def _gate_tomography(
    nqubits: int,
    gate: gates.Gate = None,
    nshots: int = int(1e4),
    ct_qubits=None,
    noise_model=None,
    backend=None,
    transpiler=None,
):
    """Runs gate tomography for a 1 or 2 qubit gate.

    It obtains a :math:`4^{n} \\times 4^{n}` matrix, where :math:`n` is the number of qubits.
    This matrix needs to be post-processed to get the Pauli-Liouville representation of the gate.
    The matrix has elements :math:`\\text{tr}(M_{j} \\, \\rho_{k})` or
    :math:`\\text{tr}(M_{j} \\, O_{l} \\rho_{k})`, depending on whether the gate
    :math:`O_{l}` is present or not.

    Args:
        nqubits (int): number of qubits of the gate.
        gate (Union[qibo.gates.Gate, list[qibo.gates.Gate]], optional):
            Gate to perform gate tomography on. Supported configurations are:
                - A single single-qubit gate.
                - A single two-qubit gate.
                - Two single-qubit gates, one applied to each qubit register.
            If ``None``, gate set tomography will be performed on an empty circuit.
            Defaults to ``None``.
        nshots (int, optional): number of shots used.
        ct_qubits (list): list specifying the [control_qubit, target_qubit]
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied to simulate
            noisy computations.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
    Returns:
        ndarray: Matrix approximating the input gate.
    """

    # Check if gate is 1 or 2 qubit gate.
    _check_nqubits(nqubits)

    backend = _check_backend(backend)

    if gate is not None:
        if isinstance(gate, gates.Gate):
            gate = [gate]
        if len(gate) == 1:
            _gate = gate[0]
            if nqubits != len(_gate.qubits):
                raise_error(
                    ValueError,
                    f"Mismatched inputs: nqubits given as {nqubits}. {_gate} is a {len(_gate.qubits)}-qubit gate.",
                )
        elif len(gate) > 2:
            raise_error(
                ValueError,
                f"Mismatched inputs: number of gates in gate = {len(gate)}. Supported configurations for _gates in gate are (1) single 1-qubit gate, (2) single 2-qubit gate, (3) two 1-qubit gates applied to each qubit register.",
            )

    # GST for empty circuit or with gates
    matrix_jk = 1j * np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):

        # additional_qubits = 0 if ancilla is None else (1 if ancilla in (0, 1) else 2)
        # circ = Circuit(nqubits + additional_qubits, density_matrix=True)
        circ = Circuit(nqubits, density_matrix=True, wire_names=ct_qubits)

        circ.add(_prepare_state(k, nqubits))

        if gate is not None:
            for _gate in gate:
                circ.add(_gate)

        for j in range(4**nqubits):
            if j == 0:
                exp_val = 1.0
            else:
                new_circ = circ.copy()
                measurements = _measurement_basis(j, nqubits)
                new_circ.add(measurements)
                observable = _get_observable(j, nqubits)
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                # print("k=%d, j=%d" %(k, j))
                # print(new_circ)
                if transpiler is not None:
                    new_circ, _ = transpiler(new_circ)
                exp_val = observable.expectation_from_samples(
                    backend.execute_circuit(new_circ, nshots=nshots).frequencies()
                )
            matrix_jk[j, k] = exp_val
    return backend.cast(matrix_jk, dtype=matrix_jk.dtype)


def GST(
    gate_set: Union[tuple, set, list],
    nqubits=None,
    nshots=int(1e4),
    noise_model=None,
    include_empty=False,
    pauli_liouville=False,
    gauge_matrix=None,
    control_qubit=None,
    target_qubit=None,
    backend=None,
    transpiler=None,
):
    """Run Gate Set Tomography on the input ``gate_set``.

    Example 1:
        Given the following ``gate_set``: ``gate_set = [(gates.RX, [np.pi/3]), gates.Z,
            (gates.PRX, [np.pi/2, np.pi/3]), (gates.GPI, [np.pi/7]), (gates.Unitary,
            [np.array([[1, 0], [0, 1]])]), gates.CNOT]``, one can can simply run GST to extract
            calibration matrices for 1- and 2-qubits (``g_1q`` and ``g_2q`` respectively):
            ``` python
            g_1q, g_2q, *gates_GST = GST(gate_set=gate_set,
                                         nshots=int(1e4),
                                         include_empty=True,
                                         backend=NumpyBackend(),
                                         )
            ```
    Other examples:
        To include 2 examples for 1qb & 2qb basis operation when probabilistic error cancellation
        is ready.
    Args:
        gate_set (tuple or set or list): set of :class:`qibo.gates.Gate` and parameters to run
            GST on. For instance, ``gate_set = [(gates.RX, [np.pi/3]), gates.Z, (gates.PRX,
            [np.pi/2, np.pi/3]), (gates.GPI, [np.pi/7]), (gates.Unitary,
            [np.array([[1, 0], [0, 1]])]), gates.CNOT]``.
        nshots (int, optional): number of shots used in Gate Set Tomography per gate.
            Defaults to :math:`10^{4}`.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied to simulate
            noisy computations.
        include_empty (bool, optional): if ``True``, additionally performs gate set tomography
            for :math:`1`- and :math:`2`-qubit empty circuits, returning the corresponding empty
            matrices in the first and second position of the ouput list.
        pauli_liouville (bool, optional): if ``True``, returns the matrices in the
            Pauli-Liouville representation. Defaults to ``False``.
        gauge_matrix (ndarray, optional): gauge matrix transformation to the Pauli-Liouville
            representation. Defaults to

            .. math::
                \\begin{pmatrix}
                    1 & 1 & 1 & 1 \\\\
                    0 & 0 & 1 & 0 \\\\
                    0 & 0 & 0 & 1 \\\\
                    1 & -1 & 0 & 0 \\\\
                \\end{pmatrix}

        control_qubit (int, optional): If gate is None & nqubits==2, do GST on this qubit & target qubit.
        target_qubit (int, optional): If gate is None & nqubits==2, do GST on this qubit & control qubit.
                                      If gate is None & nqubits==1, do GST on this qubit.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
    Returns:
        List(ndarray): input ``gate_set`` represented by matrices estimaded via GST.
    """

    if target_qubit is None and control_qubit is None:
        target_qubit = 1
        control_qubit = 0

    backend = _check_backend(backend)
    # if backend.name == "qibolab" and transpiler is None:  # pragma: no cover
    #     transpiler = Passes(
    #         connectivity=backend.platform.topology,
    #         passes=[
    #             Preprocessing(backend.platform.topology),
    #             Random(backend.platform.topology),
    #             Sabre(backend.platform.topology),
    #             Unroller(NativeGates.default()),
    #         ],
    #     )

    timings = []
    matrices = []
    empty_matrices = []
    if include_empty or pauli_liouville:
        # for nqubits in SUPPORTED_NQUBITS:
        if nqubits not in SUPPORTED_NQUBITS:
            raise_error(
                ValueError,
                f"Mismatched inputs: nqubits given as {nqubits}. GST only supports 1 and 2 qubits.",
            )
        if nqubits == 1:
            ct_qubits = [target_qubit]
        elif nqubits == 2:
            ct_qubits = [control_qubit, target_qubit]
        else:
            raise_error(
                ValueError,
                f"Input for nqubits is required. nqubits given as {nqubits}.",
            )
        tic = timeit.default_timer()
        empty_matrix = _gate_tomography(
            nqubits=nqubits,
            gate=None,
            nshots=nshots,
            ct_qubits=ct_qubits,
            noise_model=noise_model,
            backend=backend,
            transpiler=transpiler,
        )
        toc = timeit.default_timer() - tic
        empty_matrices.append(empty_matrix)
        timings.append(toc)

    if gate_set is not None:
        for _gate in gate_set:
            if _gate is not None:
                _gate, nqubits = _extract_gate(_gate)
                gate = [_gate]
            if nqubits == 1:
                ct_qubits = [target_qubit]
            elif nqubits == 2:
                ct_qubits = [control_qubit, target_qubit]

            tic = timeit.default_timer()
            matrices.append(
                _gate_tomography(
                    nqubits=nqubits,
                    gate=gate,
                    nshots=nshots,
                    ct_qubits=ct_qubits,
                    noise_model=noise_model,
                    backend=backend,
                    transpiler=transpiler,
                )
            )
            toc = timeit.default_timer() - tic
            timings.append(toc)

    if pauli_liouville:
        if gauge_matrix is not None:
            if np.linalg.det(gauge_matrix) == 0:
                raise_error(ValueError, "Matrix is not invertible")
        else:
            gauge_matrix = backend.cast(
                [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]]
            )
        PL_matrices = []
        gauge_matrix_1q = gauge_matrix
        gauge_matrix_2q = backend.np.kron(gauge_matrix, gauge_matrix)
        for matrix in matrices:
            gauge_matrix = gauge_matrix_1q if matrix.shape[0] == 4 else gauge_matrix_2q
            empty = empty_matrices[0] if matrix.shape[0] == 4 else empty_matrices[1]
            PL_matrices.append(
                gauge_matrix
                @ backend.np.linalg.inv(empty)
                @ matrix
                @ backend.np.linalg.inv(gauge_matrix)
            )
        matrices = PL_matrices

    if include_empty:
        matrices = empty_matrices + matrices

    return matrices, timings


def compute_noisy_and_noiseless_PTM(gjk=None, O_tilde=None, O_gate=None):
    """Computes the inverse noise for 1 qubit operator.

    Args:
        gjk (numpy.array): Matrix with elements :math:`\\text{tr}(Q_{j} \\, \\rho_{k})`
        O_tilde (numpy.array): :math:`\\text{tr}(Q_{j} \\, O_{l} \\rho_{k})`
            where :math:`O_{l}` is the l-th gate in the original circuit.
        O_gate (tuple or set or list): single :class:`qibo.gates.Gate` and parameters of
            the gate. For instance, gate could be any within
            ``gate_set = [(gates.RX, [np.pi/3]), gates.Z, (gates.PRX, [np.pi/2, np.pi/3]),
            (gates.GPI, [np.pi/7]), (gates.Unitary, [np.array([[1, 0], [0, 1]])]), gates.CNOT]``
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        O_hat (numpy.ndarray): Noisy representation of the gate given in PTM notation.
        O_exact_PTM (numpy.ndarray): Exact (noiseless) representation of the gate given in PTM notation.
        # ndarray: array representing the inverse noise occurring after the operator.
    """

    if gjk is None:
        raise TypeError(f"Expected gjk input")
    else:
        if np.shape(gjk) == (4, 4):
            nqubits = 1
        elif np.shape(gjk) == (16, 16):
            nqubits = 2
        else:
            raise_error(
                ValueError,
                f"Received gjk shape {np.shape(gjk)}. Expected square matrix.",
            )

    if O_tilde is None:
        raise TypeError(f"Expected O_tilde input")

    if O_gate is None:
        raise TypeError(f"Expected a `qibo.gates.Gate` gate with/without parameters.")

    # Check if gjk is same shape as O_tilde
    if np.shape(gjk) != np.shape(O_tilde):
        raise_error(
            ValueError,
            f"Received gjk shape {np.shape(gjk)}, O_tilde shape {np.shape(O_tilde)}. Expected equal dimensions.",
        )

    # Extract exact matrix form of the gate
    gate = O_gate
    if isinstance(gate, tuple):
        angles = ["theta", "phi", "lam", "unitary"]
        gate, params = gate
        params = [params] if isinstance(params[0], np.ndarray) else params
        init_args = signature(gate).parameters
        valid_angles = [arg for arg in init_args if arg in angles]
        angle_values = dict(zip(valid_angles, params))
    else:
        angle_values = {}
        init_args = signature(gate).parameters

    if "q" in init_args:
        nqubits = 1
    elif "q0" in init_args and "q1" in init_args and "q2" not in init_args:
        nqubits = 2
    else:
        raise_error(
            RuntimeError,
            f"Gate {gate} is not supported for `GST`, only 1- and 2-qubit gates are supported.",
        )

    if "unitary" in angle_values:
        gate = gate(angle_values["unitary"][0], *range(nqubits))
    else:
        gate = gate(*range(nqubits), **angle_values)

    O_gate_matrix = gate.matrix()

    # Set up preliminaries for computation of O_hat
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Pauligates_1qubit = [
        gates.I(0).matrix(),
        gates.X(0).matrix(),
        gates.Y(0).matrix(),
        gates.Z(0).matrix(),
    ]
    Pauligates_2qubit = []
    for ii in range(0, 4):
        for jj in range(0, 4):
            temp_matrix = np.kron(Pauligates_1qubit[ii], Pauligates_1qubit[jj])
            Pauligates_2qubit.append(temp_matrix)

    # Compute operator_hat
    # $\mathcal{\hat{O}}^{(l)} = T g^{-1} \mathcal{\tilde{O}}^{(l)} T^{-1}$
    O_hat = np.zeros((4**nqubits, 4**nqubits))
    if nqubits == 1:
        O_hat = T * np.linalg.inv(gjk) * O_tilde * np.linalg.inv(T)
    elif nqubits == 2:
        O_hat = (
            np.kron(T, T) * np.linalg.inv(gjk) * O_tilde * np.linalg.inv(np.kron(T, T))
        )

    # Exact PTM of operator(s)
    # $\mathcal{{O}}_{\sigma, \tau}^{(l), exact} = \frac{1}{d} tr(\sigma \mathcal(O) \tau)$ (exact PTM of the operator(s)
    O_exact_PTM = np.zeros((4**nqubits, 4**nqubits))
    for ii in range(0, 4**nqubits):
        for jj in range(0, 4**nqubits):
            if nqubits == 1:
                O_exact_PTM[ii, jj] = (1 / 2**nqubits) * np.trace(
                    Pauligates_1qubit[ii]
                    @ O_gate_matrix
                    @ Pauligates_1qubit[jj]
                    @ np.conjugate(np.transpose(O_gate_matrix))
                )
            elif nqubits == 2:
                O_exact_PTM[ii, jj] = (1 / 2**nqubits) * np.trace(
                    Pauligates_2qubit[ii]
                    @ O_gate_matrix
                    @ Pauligates_2qubit[jj]
                    @ np.conjugate(np.transpose(O_gate_matrix))
                )

    # Compute inverse noise
    # $(\mathcal{N}^{(l)})^{-1} = \mathcal{{O}}^{(l), exact} (\mathcal{\hat{O}}^{(l)})^{-1}$
    # invNoise = np.zeros((4**nqubits, 4**nqubits))
    # invNoise = np.array(O_exact_PTM) @ (np.array(np.linalg.inv(O_hat)))

    return np.array(O_hat), np.array(O_exact_PTM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20"],
        default="numpy",
        help="Execution device",
    )
    parser.add_argument("--nshots", type=int, default=int(1e4), help="Number of shots")
    args = parser.parse_args()

    backend = construct_backend("numpy")  # , platform="sinq-20")

    ### Determine qubits ###
    single_qubit_indices = [0, 1]
    # single_qubit_indices = np.arange(0, 20, 1)
    two_qubit_pairs = [[0, 1]]
    # two_qubit_pairs = [[0, 1], [0, 3], [1, 4], [2, 3], [2, 7], [3, 4], [3, 8], [4, 5], [4, 9], [5, 6], [5, 10], [6, 11], [7, 8], [7, 12], [8, 9], [8, 13], [9, 10], [9, 14], [10, 11], [10, 15], [11, 16], [12, 13], [13, 14], [13, 17], [14, 15], [14, 18], [15, 16], [15, 19], [17, 18], [18, 19]]

    from qibo.transpiler import NativeGates, Passes, Unroller

    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)

    ### GST EMPTY 1 QUBIT ###
    empty_1qb_matrices = []
    empty_1qb_timings = []
    for idx in single_qubit_indices:
        logging.info(f"Empty 1qb matrix on qubit {idx}")
        empty_1qb_matrix, timings = GST(
            gate_set=None,
            nqubits=1,
            nshots=args.nshots,
            include_empty=True,
            control_qubit=None,
            target_qubit=int(idx),
            noise_model=None,
            backend=backend,
            transpiler=custom_pipeline,
        )
        empty_1qb_matrices.append(empty_1qb_matrix)
        empty_1qb_timings.append(timings)

    ### GST EMPTY 2 QUBIT ###
    empty_2qb_matrices = []
    empty_2qb_timings = []
    for pair in two_qubit_pairs:
        logging.info(f"Empty 2qb matrix on qubits {pair[0]}_{pair[1]}")
        empty_2qb_matrix, timings = GST(
            gate_set=None,
            nqubits=2,
            nshots=args.nshots,
            include_empty=True,
            control_qubit=int(pair[0]),
            target_qubit=int(pair[1]),
            noise_model=None,
            backend=backend,
            transpiler=custom_pipeline,
        )
        empty_2qb_matrices.append(empty_2qb_matrix)
        empty_2qb_timings.append(timings)

    ### GST 1 QUBIT GATE ###
    gate_set_1qb = [(gates.GPI2, [np.pi / 4]), (gates.GPI2, [np.pi / 3])]
    gates_1qb_matrices = []
    gates_1qb_timings = []
    for idx in single_qubit_indices:
        logging.info(f"1qb gate on qubits {idx}")
        gates_1qb_matrix, timings = GST(
            gate_set=gate_set_1qb,
            nqubits=1,
            nshots=args.nshots,
            include_empty=False,
            control_qubit=None,
            target_qubit=int(idx),
            noise_model=None,
            backend=backend,
            transpiler=custom_pipeline,
        )
        gates_1qb_matrices.append(gates_1qb_matrix)
        gates_1qb_timings.append(timings)

    ### GST 2 QUBIT GATE ###
    gate_set_2qb = [gates.CZ]
    gates_2qb_matrices = []
    gates_2qb_timings = []
    for pair in two_qubit_pairs:
        logging.info(f"2qb gate on qubits {pair[0]}_{pair[1]}")
        gates_2qb_matrix, timings = GST(
            gate_set=gate_set_2qb,
            nqubits=2,
            nshots=args.nshots,
            include_empty=False,
            control_qubit=int(pair[0]),
            target_qubit=int(pair[1]),
            noise_model=None,
            backend=backend,
            transpiler=custom_pipeline,
        )
        gates_2qb_matrices.append(gates_2qb_matrix)
        gates_2qb_timings.append(timings)

    ### Compute PTM for 1 QUBIT  ###
    noisyPTM_1qb = []
    noiselessPTM_1qb = []
    for ii in range(0, len(empty_1qb_matrices)):
        gjk_1q = empty_1qb_matrices[ii][0]

        temp_noisyPTM_1qb = []
        temp_noiselessPTM_1qb = []
        for jj in range(len(gates_1qb_matrices[ii])):
            O_tilde = gates_1qb_matrices[ii][jj]
            temp1, temp2 = compute_noisy_and_noiseless_PTM(
                gjk=gjk_1q, O_tilde=O_tilde, O_gate=gate_set_1qb[jj]
            )
            temp_noisyPTM_1qb.append(temp1)
            temp_noiselessPTM_1qb.append(temp2)

        noisyPTM_1qb.append(temp_noisyPTM_1qb)
        noiselessPTM_1qb.append(temp_noiselessPTM_1qb)

    ### Compute PTM for 2 QUBIT ###
    noisyPTM_2qb = []
    noiselessPTM_2qb = []
    for ii in range(0, len(empty_2qb_matrices)):
        gjk_2q = empty_2qb_matrices[ii][0]
        O_tilde = gates_2qb_matrices[ii][0]
        temp1, temp2 = compute_noisy_and_noiseless_PTM(
            gjk=gjk_2q, O_tilde=O_tilde, O_gate=gate_set_2qb[0]
        )
        noisyPTM_2qb.append(temp1)
        noiselessPTM_2qb.append(temp2)

    ### Compute norm of the difference between NOISY and NOISELESS ###

    result_dict = {}
    result_dict["oneQubitNorms"] = {}
    result_dict["twoQubitNorms"] = {}

    ### One qubit ###
    for ii in range(0, len(empty_1qb_matrices)):
        qubit = single_qubit_indices[ii]
        logging.info(f"Compute norm of 1qb matrices on qubits {qubit}")
        result_dict["oneQubitNorms"][f"{qubit}"] = {}
        for jj in range(0, len(gate_set_1qb)):
            if isinstance(gate_set_1qb[jj], tuple):
                gate, params = gate_set_1qb[jj]
            else:
                gate = gate_set_1qb[jj]
                params = []
            gate = gate(qubit, *params)
            gate_name = gate.name

            diff = noisyPTM_1qb[ii][jj] - noiselessPTM_1qb[ii][jj]
            one_norm = norm(diff, ord=1)
            inf_norm = norm(diff, ord=np.inf)
            two_norm = norm(diff, ord=2)
            result_dict["oneQubitNorms"][f"{qubit}"][
                f"{gate_name}({qubit}, {params})"
            ] = {}
            result_dict["oneQubitNorms"][f"{qubit}"][f"{gate_name}({qubit}, {params})"][
                "one_norm"
            ] = one_norm
            result_dict["oneQubitNorms"][f"{qubit}"][f"{gate_name}({qubit}, {params})"][
                "inf_norm"
            ] = inf_norm
            result_dict["oneQubitNorms"][f"{qubit}"][f"{gate_name}({qubit}, {params})"][
                "two_norm"
            ] = two_norm
            result_dict["oneQubitNorms"][f"{qubit}"][f"{gate_name}({qubit}, {params})"][
                "time (s)"
            ] = gates_1qb_timings[ii][jj]

    ### Two qubit ###
    for ii in range(0, len(empty_2qb_matrices)):
        pair = two_qubit_pairs[ii]
        logging.info(f"Compute norm of 2qb matrices on qubits {pair[0]}_{pair[1]}")

        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"] = {}

        if isinstance(gate_set_2qb[0], tuple):
            gate, params = gate_set_1qb[0]
        else:
            gate = gate_set_2qb[0]
            params = []
        gate = gate(pair[0], pair[1], *params)
        gate_name = gate.name

        diff = noisyPTM_2qb[ii][jj] - noiselessPTM_2qb[ii][jj]
        one_norm = norm(diff, ord=1)
        inf_norm = norm(diff, ord=np.inf)
        two_norm = norm(diff, ord=2)
        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"][
            f"{gate_name}({pair[0]}, {pair[1]}, {params})"
        ] = {}
        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"][
            f"{gate_name}({pair[0]}, {pair[1]}, {params})"
        ]["one_norm"] = one_norm
        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"][
            f"{gate_name}({pair[0]}, {pair[1]}, {params})"
        ]["inf_norm"] = inf_norm
        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"][
            f"{gate_name}({pair[0]}, {pair[1]}, {params})"
        ]["two_norm"] = two_norm
        result_dict["twoQubitNorms"][f"{pair[0]}_{pair[1]}"][
            f"{gate_name}({pair[0]}, {pair[1]}, {params})"
        ]["time (s)"] = gates_2qb_timings[ii][0]

    # SAVE JSON under data/<scriptname>/<device>/results.json
    out_dir = config.output_dir_for(__file__) / args.device
    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, "results.json"), "w", encoding="utf-8"
    ) as json_file:
        json.dump(result_dict, json_file, indent=4)

    # SAVE NPY FILES under data/<scriptname>/<device>/matrices/
    matrices_dir = os.path.join(out_dir, "matrices")
    os.makedirs(matrices_dir, exist_ok=True)
    for ii in range(0, len(empty_1qb_matrices)):
        qubit = single_qubit_indices[ii]
        np.save(
            os.path.join(matrices_dir, f"empty_1qb_matrix_qubit{qubit}.npy"),
            empty_1qb_matrices[ii],
        )
    for ii in range(0, len(empty_2qb_matrices)):
        pair = two_qubit_pairs[ii]
        np.save(
            os.path.join(
                matrices_dir, f"empty_2qb_matrix_qubits{pair[0]}_{pair[1]}.npy"
            ),
            empty_2qb_matrices[ii],
        )
    for ii in range(0, len(gates_1qb_matrices)):
        for jj in range(0, len(gates_1qb_matrices[0])):
            qubit = single_qubit_indices[ii]
            gate = gate_set_1qb[jj]
            if isinstance(gate_set_1qb[jj], tuple):
                gate, params = gate_set_1qb[jj]
            else:
                gate = gate_set_1qb[jj]
                params = []
            gate = gate(qubit, *params)
            gate_name = gate.name
            np.save(
                os.path.join(
                    matrices_dir,
                    f"gate_{gate_name}_{np.around(params,4)}_qubit{qubit}.npy",
                ),
                gates_1qb_matrices[ii][jj],
            )
    for ii in range(0, len(empty_2qb_matrices)):
        pair = two_qubit_pairs[ii]

        if isinstance(gate_set_2qb[0], tuple):
            gate, params = gate_set_1qb[0]
        else:
            gate = gate_set_2qb[0]
            params = []
        gate = gate(pair[0], pair[1], *params)
        gate_name = gate.name
        np.save(
            os.path.join(
                matrices_dir,
                f"gate_{gate_name}_{np.around(params,4)}_qubits{pair[0]}_{pair[1]}.npy",
            ),
            gates_2qb_matrices[ii],
        )
