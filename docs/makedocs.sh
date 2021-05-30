sphinx-build -b html . _build/html
sphinx-build -b latex . _build/latex
cd _build/latex
xelatex nmr-espy.tex
xelatex nmr-espy.tex
