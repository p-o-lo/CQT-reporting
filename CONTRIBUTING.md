# Experiment Contribution Guidelines

This are the guidelines to create a new `<experiment>` for the benchmarking suite. 
This project organizes experiments as subfolders under `scripts/<experiment>/main.py` and collects results under `data/<experiment>/`. Follow these standards so your contribution runs via the batch runner and integrates with plotting/report generation.

## Rules

- Directory: `scripts/<experiment>/`
- Entrypoint: `scripts/<experiment>/main.py`
- Data output: `data/<experiment>/<device>/results.json`
  - `device` is provided via `--device` and must be either `numpy` or `nqch`.
  - Create the output directory with `mkdir(parents=True, exist_ok=True)` before writing.
- Use the helper in `scripts/config.py`:
  - `from pathlib import Path` (already available)
  - `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import config`
  - Resolve the experiment folder and device path via:
    ```python
    out_dir = config.output_dir_for(__file__) / args.device
    out_dir.mkdir(parents=True, exist_ok=True)
    ```
- Optional extra artifacts (e.g., plots, params, matrices) should be saved inside the same folder: `data/<experiment>/<device>/...`.
- Provide `argparse` with sensible defaults; do not override CLI args in code.
- Ensure the script runs when the subfolder is added to `scripts/runscripts.py`.
- Write outputs under `data/<experiment>/...`. Make sure to use a robust path handling and `mkdir(parents=True, exist_ok=True)`.
- For histogram-like results, include all bitstrings (zero for missing) in the frequencies dict.
- If using a device option, document that `"numpy"` is supported for local simulation.
- This is an example of JSON that we can plot: 

```{"success_rate": 1.0, "plotparameters": {"frequencies": {"00000": 508, "00001": 0, "00010": 0, "00011": 0, "00100": 0, "00101": 0, "00110": 0, "00111": 0, "01000": 0, "01001": 0, "01010": 0, "01011": 0, "01100": 0, "01101": 0, "01110": 0, "01111": 0, "10000": 0, "10001": 0, "10010": 0, "10011": 0, "10100": 0, "10101": 0, "10110": 0, "10111": 0, "11000": 0, "11001": 0, "11010": 0, "11011": 0, "11100": 0, "11101": 0, "11110": 0, "11111": 492}}}```


## Devices and execution

- Document `"numpy"` as a supported device for local simulation:
  - Example help: `help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)"`
- If using remote execution, load credentials via Dynaconf (`.secrets.toml`) and keep defaults runnable locally.

- Device selection:
  - `--device` must be one of `numpy` or `nqch`.
  - For local simulation use `numpy` (e.g., `qibo.set_backend("numpy")` or `construct_backend("numpy")`).
  - For hardware/cloud (`nqch`), load credentials via Dynaconf (`.secrets.toml`) when needed.
- Runner:
  - Run from the repository root. Example: `make runscripts`
  - The batch runner calls `python scripts/<experiment>/main.py` with defaults unless otherwise configured.


## Do not

- Do not hardcode overrides of parsed CLI arguments.
- Do not write outputs outside `data/<experiment>` or without ensuring directories exist.
- Do not output non-JSON data. 
- Do not mix experiment and plot generation
- We do not use or require the environment variable `QIBOLAB_PLATFORMS`. Do not set it in new experiments.


## CLI requirements

Every `main.py` must accept at least:
- `--device {numpy,nqch}` (default: `numpy`)
- Other arguments as needed (e.g., `--nshots`, qubit lists, etc.)



## Good code snippets

Argparse and respecting CLI parameters:
```python
import argparse
# ...existing code...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["numpy", "nqch"], default="numpy",
                        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)")
    parser.add_argument("--nshots", type=int, default=1000, help="Number of shots for each circuit")
    # add other args as needed
    args = parser.parse_args()
    main(**vars(args))
```

## Writing outputs

Write a single JSON with computed outputs to `data/<experiment>/<device>/results.json`.

Example:
```python
# ...existing code building `results`...
out_dir = config.output_dir_for(__file__) / args.device
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
```

Optionally, you may also write a `data.json` with static metadata to the same folder.

## Frequencies and histograms

- For histogram-like results, include all bitstrings (use zero for missing counts) in the frequencies dict.
- Example for a 2-qubit measurement:
```python
num_bits = 2
all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
results["plotparameters"]["frequencies"]["[0, 1]"] = prob_dict
```

Simple GHZ experiment writing to `data/ghz` (from `scripts/GHZ/main.py`):
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config
out_dir = config.output_dir_for(__file__) / args.device
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "results.json").open("w") as f:
    json.dump(results, f, indent=4)
```

Local simulation:
```python
from qibo import set_backend
if args.device == "numpy":
    set_backend("numpy")
```

## Remote execution (nqch) example

Example snippet for running on nqch using Dynaconf and qibo_client. Save results to data/<experiment>/<device>/results.json via scripts/config.py.

```python
import argparse, json, sys, pathlib
from qibo import Circuit, gates, set_backend
from dynaconf import Dynaconf
import qibo_client

# Make scripts/config.py importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config

def build_bell():
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.M(0, 1))
    return c

def main(nshots, device):
    c = build_bell()
    if device == "nqch":
        settings = Dynaconf(settings_files=[".secrets.toml"], environments=True, env="default")
        client = qibo_client.Client(token=settings.key)
        job = client.run_circuit(c, device="nqch", nshots=nshots)
        r = job.result(verbose=True)
        freq = r.frequencies()
    else:
        set_backend("numpy")
        r = c(nshots=nshots)
        freq = r.frequencies()

    # Minimal results payload
    results = {"plotparameters": {"frequencies": freq}, "nshots": nshots, "device": device}

    out_dir = config.output_dir_for(__file__) / device
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["numpy", "nqch"], default="numpy")
    parser.add_argument("--nshots", type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))
```

