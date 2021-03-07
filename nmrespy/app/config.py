import os
import nmrespy

# useful paths to various directories
NMRESPYDIR = os.path.dirname(nmrespy.__file__)
GUIDIR = os.path.join(NMRESPYDIR, 'gui')
TMPDIR = os.path.join(GUIDIR, 'tmp')
IMAGESDIR = os.path.join(NMRESPYDIR, 'images')

# web links
MFGROUPLINK = 'http://foroozandeh.chem.ox.ac.uk/home'
NMRESPYLINK = 'https://nmr-espy.readthedocs.io/en/latest/index.html'
GUIDOCLINK = 'https://nmr-espy.readthedocs.io/en/latest/gui.html'
EMAILLINK = r"mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query"

# GUI font
MAINFONT = 'Helvetica'

# colors related to plot
BGCOLOR = '#e4eaef'
PLOTCOLOR = '#ffffff' # plot background
REGIONCOLOR = '#7fd47f' # region rectangle patch
NOISEREGIONCOLOR = '#66b3ff' # noise region rectange patch
PIVOTCOLOR = '#ff0000' # pivot line plot
NOTEBOOKCOLOR = '#c4d1dc'
ACTIVETABCOLOR = '#648ba4'
BUTTONGREEN = '#9eda88'
BUTTONORANGE = '#ffb861'
BUTTONRED = '#ff9894'
BUTTONDEFAULT = '#6699cc'
READONLYENTRYCOLOR = '#cde6ff'
