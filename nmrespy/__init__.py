from pathlib import Path
from ._version import __version__

NMRESPYPATH = Path(__file__).parent
IMAGESPATH = NMRESPYPATH / 'images'
MFLOGOPATH = IMAGESPATH / 'mf_logo.png'
NMRESPYLOGOPATH = IMAGESPATH / 'nmrespy_full.png'
BOOKICONPATH = IMAGESPATH / 'book_icon.png'
GITHUBLOGOPATH = IMAGESPATH / 'github.png'
EMAILICONPATH = IMAGESPATH / 'email_icon.png'

# To assist users with manual install of TopSpin GUI loader
TOPSPINPATH = NMRESPYPATH / 'app/_topspin.py'

GITHUBLINK = 'https://github.com/foroozandehgroup/NMR-EsPy'
MFGROUPLINK = 'http://foroozandeh.chem.ox.ac.uk/home'
DOCSLINK = 'https://foroozandehgroup.github.io/NMR-EsPy'
MAILTOLINK = r'mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query'
