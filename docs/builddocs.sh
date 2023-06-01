#!/bin/bash
personal_pcs="precision spectre"
work_pcs="parsley.chem.ox.ac.uk belladonna.chem.ox.ac.uk"
found=1
for pc in $personal_pcs
do
    if [ $HOSTNAME = $pc ] ; then
        NMRESPYPATH=/home/simon/Documents/DPhil/projects/NMR-EsPy
        found=0
    fi
done

if [ $found = 1 ]; then
    for pc in $work_pcs
    do
        echo $pc
        if [ $HOSTNAME = $pc ] ; then
            NMRESPYPATH=/u/mf/jesu2901/DPhil/projects/spectral_estimation/NMR-EsPy
            found=0
        fi
    done
fi

if [ $found = 1 ]; then
    echo "Unknown PC: add to the script."
    exit
fi

cd $NMRESPYPATH/docs
$NMRESPYPATH/.venv/bin/sphinx-build -b html . ./_build/html
xdotool key "Super_L+Right" && xdotool key "Ctrl+r" && xdotool key "Super_L+Left"
$NMRESPYPATH/.venv/bin/sphinx-build -b latex . ./_build/latex
python latex_fudge.py
cd ./_build/latex/
xelatex --shell-escape nmr-espy.tex
xelatex --shell-escape nmr-espy.tex
