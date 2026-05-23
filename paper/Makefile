
PDFS = main.pdf

.PHONY: all clean

all: main.pdf

INCLUDES = include/*.tex

PARTS = parts/*.tex

MAIN = main.tex

%.pdf: $(MAIN) $(PARTS) $(INCLUDES)
	pdflatex -interaction=nonstopmode $*
	bibtex $*
	pdflatex -interaction=nonstopmode $*
	pdflatex -interaction=nonstopmode $*

force:
	touch $(MAIN) && $(MAKE) all

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
	rm -rf latex.out
