TARGET = report.pdf

.PHONY: build clean pdf

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build:
	python src/main.py --experiment-dir $(EXPERIMENT_DIR)

pdf: build
	mkdir -p build
	pdflatex -output-directory=build report.tex
	cp build/report.pdf .

clean:
	rm -f build/* 

all: pdf