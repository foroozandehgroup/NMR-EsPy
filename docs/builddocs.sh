#!/bin/bash
if [ $HOSTNAME = "precision" ] ; then
	NMRESPYDIR=/home/simon/Documents/DPhil/projects/spectral_estimation/NMR-EsPy
elif [ $HOSTNAME = "belladonna.chem.ox.ac.uk"  ] ; then
	NMRESPYDIR=/u/mf/jesu2901/Documents/DPhil/projects/spectral_estimation/NMR-EsPy
fi	

cd $NMRESPYDIR/docs
$NMRESPYDIR/.venv/bin/sphinx-build -b html . ./build/html
$NMRESPYDIR/.venv/bin/sphinx-build -b latex . ./build/latex
python3 latex_fudge.py
cd ./build/latex/
xelatex nmr-espy.tex
xelatex nmr-espy.tex