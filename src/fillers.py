from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Dict, Any, List


class Filler(ABC):
    """Abstract base class for filling sections of the PDF report."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.data_path = Path("data") / experiment_dir

    @abstractmethod
    def get_section_name(self) -> str:
        """Return the name of the section this filler handles."""
        pass

    @abstractmethod
    def fill_data(self) -> Dict[str, Any]:
        """Extract and process data for this section."""
        pass

    @abstractmethod
    def get_template_context(self) -> Dict[str, Any]:
        """Return template context variables for rendering."""
        pass


class LibraryVersionsFiller(Filler):
    """Filler for the library versions summary table."""

    def get_section_name(self) -> str:
        return "Library Versions"

    def fill_data(self) -> Dict[str, Any]:
        """Extract library version data from experiment files."""
        # Look for metadata or configuration files that might contain library versions
        version_data = {}

        # Check for common metadata files
        metadata_files = [
            "metadata.json",
            "versions.json",
            "requirements.json",
            "environment.json",
        ]

        for filename in metadata_files:
            file_path = self.data_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if "libraries" in data:
                            version_data = data["libraries"]
                        elif "versions" in data:
                            version_data = data["versions"]
                        elif isinstance(data, dict) and any(
                            key in data
                            for key in ["qibo", "numpy", "qibolab", "qibocal"]
                        ):
                            version_data = data
                        break
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        # Default versions if no metadata found
        if not version_data:
            version_data = {
                "qibo": "0.2.18",
                "numpy": "2.3.0",
                "qibolab": "0.2.7",
                "qibocal": "0.2.2",
            }
            print(
                f"No library version metadata found in {self.data_path}, using defaults"
            )

        return version_data

    def get_template_context(self) -> Dict[str, Any]:
        """Return template context for library versions section."""
        library_versions = self.fill_data()

        # Convert to table format
        version_headers = ["Library", "Version"]
        version_table = []

        for library, version in library_versions.items():
            version_table.append([library, str(version)])

        return {
            "library_versions_headers": version_headers,
            "library_versions_table": version_table,
            "has_library_versions": len(version_table) > 0,
        }
