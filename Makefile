TARGET = report.pdf

.PHONY: build clean pdf runscripts

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build: clean
	@echo "Building latex report..."
	python src/main.py --experiment-dir $(EXPERIMENT_DIR)

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

all: pdf