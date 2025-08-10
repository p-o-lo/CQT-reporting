# Experiment Contribution Guidelines

This are the guidelines to create a new `<experiment>` for the benchmarking suite. 
This project organizes experiments as subfolders under `scripts/<experiment>/main.py` and collects results under `data/<experiment>/`. Follow these standards so your contribution runs via the batch runner and integrates with plotting/report generation.

## Rules

- Directory: `scripts/<experiment>/`
- Data output: `data/<experiment>/`
- Place your runnable entrypoint in: `scripts/<experiment>/main.py`.
- Add a helper in `<experiment>/plots.py` with a signature like `plot_<experiment>(data_json, output_path="build/")`
- The `if __name__ == "__main__":` should call a `main()` function that performs the experiment, then should create the plots. 
- Call the plot function only *after* running the experiments. Use as input the JSON that the main code of the experiment is creating. 
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

## Do not

- Do not hardcode overrides of parsed CLI arguments.
- Do not write outputs outside `data/<experiment>` or without ensuring directories exist.
- Do not output non-JSON data. 
- Do not mix experiment and plot generation



## Good code snippets

Argparse and respecting CLI parameters:
```python
import argparse
# ...existing code...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nqch", type=str,
                        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)")
    parser.add_argument("--nshots", default=1000, type=int, help="Number of shots for each circuit")
    # add other args as needed
    args = vars(parser.parse_args())
    main(**args)
```

Writing outputs safely to `data/<experiment>` using pathlib (from `scripts/grover2q/main.py`):
```python
from pathlib import Path
import json
# ...existing code...
repo_root = Path(__file__).resolve().parents[2]  # project root
out_dir = repo_root / "data" / "grover2q"
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "data.json").open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
with (out_dir / "results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
```

Frequencies dictionary including all bitstrings with zeros for missing outcomes (from `scripts/grover2q/main.py`):
```python
# freq = r.frequencies()
num_bits = len(qubits)
all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
prob_dict = {bs: (freq.get(bs, 0) / nshots) for bs in all_bitstrings}
results["plotparameters"]["frequencies"][f"{qubits}"] = prob_dict
```

Simple GHZ experiment writing to `data/ghz` (from `scripts/GHZ/main.py`):
```python
# ...existing code...
all_bitstrings = [format(i, f"0{nqubits}b") for i in range(2**nqubits)]
freq_dict = {bitstr: frequencies.get(bitstr, 0) for bitstr in all_bitstrings}
results = {
    "success_rate": success_rate,
    "plotparameters": {"frequencies": freq_dict},
}
output_dir = os.path.join("data", "ghz")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "ghz_5q_samples.json"), "w") as f:
    json.dump(results, f, indent=4)
```

Training-style experiment saving results (from `scripts/reuploading/main.py`):
```python
results = {
    "epoch_data": plot_data,
    "loss_history": loss_history,
    "median_predictions": median_pred.tolist(),
    "mad_predictions": mad_pred.tolist(),
}
with open(os.path.join("data/reuploading", "results_reuploading.json"), "w") as json_file:
    json.dump(results, json_file, indent=4)
```

Batch runner integration (from `scripts/runscripts.py`):
```python
subfolders = [
    # ...existing items...
    "your_experiment_name",
]
# The runner executes scripts/<subfolder>/main.py with default args.
```

## Output format suggestions

- Provide two JSONs when useful:
  - `data.json`: static metadata (e.g., input qubits, device, shots, targets).
  - `results.json`: computed outputs (e.g., success rates, plotparameters).
- For histogram-like outputs:
  - `results["plotparameters"]["frequencies"]` should be a dict keyed by bitstrings.
  - Include all bitstrings of the measured register with zero for missing keys.
  - Example:
    ```json
    {
      "success_rate": {"[0, 1]": 0.93},
      "plotparameters": {
        "frequencies": {
          "[0, 1]": { "00": 0.0, "01": 0.07, "10": 0.0, "11": 0.93 }
        }
      }
    }
    ```

