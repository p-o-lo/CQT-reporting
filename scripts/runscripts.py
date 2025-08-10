import os
import subprocess

# List of subfolders containing main.py
subfolders = [
    # "GHZ",
    # "mermin5q",
    "grover2q",
    # "tomography",
    # "reuploading",
    # "universal_approximant",
    # "process_tomography",
]

# Base path to the scripts directory
base_path = "scripts/"

# Iterate through each subfolder and call main.py
for subfolder in subfolders:
    script_path = os.path.join(base_path, subfolder, "main.py")
    if os.path.exists(script_path):
        try:
            print(f"Running {script_path}...")
            subprocess.run(["python", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script_path}: {e}")
    else:
        print(f"main.py not found in {subfolder}")
