TARGET = report.pdf

.PHONY: build clean pdf

# Default experiment directory
EXPERIMENT_DIR ?= rb-1306

build:
	python src/main.py --experiment-dir $(EXPERIMENT_DIR)

pdf: build
	cd build && pdflatex report.tex
	mv build/report.pdf ./

clean:
	rm -f build/*.tex build/*.pdf build/*.aux build/*.log build/*.out

all: pdf