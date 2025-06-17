# Quantum Benchmark Reporter

This project is designed to automate the generation of PDF reports based on benchmarking experiments conducted on quantum computers. The reports are generated using a LaTeX template that is populated with data from JSON files containing benchmark results.

## Project Structure

```
quantum-benchmark-reporter
├── src
│   ├── __init__.py
│   ├── report_generator.py
│   ├── data_processor.py
│   └── templates
│       └── report_template.tex
├── data
│   └── sample_benchmark.json
├── output
│   └── .gitkeep
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

To set up the project, you will need to create a virtual environment and install the required dependencies. You can do this using the following commands:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

# Install dependencies
uv sync
```

### Troubleshooting

If you encounter an error with `uv.lock`, delete the file and regenerate it:

```bash
# Remove the corrupted lock file
rm uv.lock

# Regenerate dependencies
uv sync
```

## Usage

1. **Prepare your benchmark data**: Place your JSON benchmark data in the `data` directory. You can use the provided `sample_benchmark.json` as a template.

2. **Generate the report**: Run the `report_generator.py` script to generate the LaTeX file populated with your benchmark data.

   ```bash
   python src/report_generator.py
   ```

3. **Build the PDF report**: Use the Makefile to compile the LaTeX file into a PDF.

   ```bash
   make
   ```

The generated PDF report will be available in the `output` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.