NMRESPYPATH=$HOME/Documents/DPhil/projects/NMR-EsPy
cd $NMRESPYPATH/docs
rm -r _build
$NMRESPYPATH/.venv/bin/sphinx-build -b html . ./_build/html
$NMRESPYPATH/.venv/bin/sphinx-build -b latex . ./_build/latex
python latex_tweaks.py
cd ./_build/latex/
xelatex --shell-escape nmr-espy.tex
xelatex --shell-escape nmr-espy.tex
