TARGET = report.pdf

.PHONY: build clean pdf runscripts runscripts-device

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build: clean
	@echo "Building latex report..."
	python src/main.py --experiment-left $(EXPERIMENT_DIR) --experiment-right BASELINE

pdf: build
	@mkdir -p build
	@echo "Compiling LaTeX report in pdf..."
	pdflatex -output-directory=build report.tex > build/pdflatex.log
	@cp build/report.pdf .

clean:
	@echo "Cleaning build directory..."
	@rm -f build/*

runscripts:
	@echo "Running scripts..."
	python3 scripts/runscripts.py

runscripts-device:
	@echo "Running scripts with device=nqch..."
	python3 scripts/runscripts.py --device nqch

all: pdf