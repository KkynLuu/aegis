all: main.pdf cover_letter.pdf

main.pdf: main.tex references.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

cover_letter.pdf: cover_letter.tex
	pdflatex -interaction=nonstopmode cover_letter.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out

.PHONY: all clean
