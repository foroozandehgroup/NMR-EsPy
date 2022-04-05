# _paths_and_links.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 11:03:06 GMT

from pathlib import Path

NMRESPYPATH = Path(__file__).parent
STYLESHEETPATH = NMRESPYPATH / "config/nmrespy_custom.mplstyle"
IMAGESPATH = NMRESPYPATH / "images"
MFLOGOPATH = IMAGESPATH / "mf_logo.png"
NMRESPYLOGOPATH = IMAGESPATH / "nmrespy_full.png"
BOOKICONPATH = IMAGESPATH / "book_icon.png"
GITHUBLOGOPATH = IMAGESPATH / "github.png"
EMAILICONPATH = IMAGESPATH / "email_icon.png"
TOPSPINPATH = NMRESPYPATH / "app/_topspin.py"
GITHUBLINK = "https://github.com/foroozandehgroup/NMR-EsPy"
MFGROUPLINK = "http://foroozandeh.chem.ox.ac.uk/home"
DOCSLINK = "https://foroozandehgroup.github.io/NMR-EsPy"
MAILTOLINK = r"mailto:simon.hulse@chem.ox.ac.uk?subject=NMR-EsPy query"
PAPERS = {
    "Newton Meets Ockham": {
        "doi": "https://doi.org/10.1016/j.jmr.2022.107173",
        "citation": (
            "Simon G. Hulse, Mohammadali Foroozandeh. \"Newton meets Ockham: "
            "Parameter estimation and model selection of NMR data with NMR-EsPy\". "
            "J. Magn. Reson. 338 (2022) 107173."
        ),
    },
}
