from pathlib import Path

# Repository root (two levels above any script in scripts/<name>/main.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def output_dir_for(script_file: str | Path) -> Path:
    """Return data/<script-dir-name>/ for the given script file."""
    script_path = Path(script_file).resolve()
    return DATA_DIR / script_path.parent.name
