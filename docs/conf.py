# conf.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 11 Dec 2023 14:21:39 EST

import os
import sys
import sphinx_nameko_theme
import nmrespy

sys.path.insert(0, os.path.abspath("exts"))
sys.path.insert(0, os.path.abspath(".."))

project = "NMR-EsPy"
author = "Simon Hulse & Mohammadali Foroozandeh"
copyright = "2023, Simon Hulse & Mohammadali Foroozandeh"
version = nmrespy.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgmath",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_selective_exclude.eager_only",
]

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
rst_prolog = """
.. raw:: html

   <style>
      .grey {color:#808080}
      .red {color:#ff0000}
      .green {color:#008000}
      .blue {color:#0000ff}
      .oscblue {color:#1063e0}
      .oscorange {color:#eb9310}
      .oscgreen {color:#2bb539}
      .oscred {color:#d4200c}
      .regiongreen {color: #7fd47f}
      .regionblue {color: #66b3ff}
   </style>

.. role:: grey
.. role:: red
.. role:: green
.. role:: blue
.. role:: oscblue
.. role:: oscorange
.. role:: oscgreen
.. role:: oscred
.. role:: regiongreen
.. role:: regionblue
"""

todo_include_todos = True

autosectionlabel_prefix_document = True
autoclass_content = "both"
autodoc_typehints = "signature"
autodoc_typehints_format = "fully-qualified"

# HTML Settings
html_static_path = ["_static"]
html_theme = "nameko"
html_theme_path = [sphinx_nameko_theme.get_html_theme_path()]
html_scaled_image_link = False
html_sidebars = {
    "**": [
        "links.html",
        "localtoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

autosummary_generate = True

latex_engine = "xelatex"
latex_elements = {
    "inputenc": "",
    "babel": "\\usepackage{polyglossia}",
    "fontenc": "",
    "maxlistdepth": "10",
    "preamble": r"""
\usepackage[math-style=ISO,bold-style=ISO]{unicode-math}
\usepackage{fontspec}
\setmainfont{FiraSans}[
Path           = ../../fonts/FiraSans/,%
Extension      = .ttf,%
UprightFont    = *-Regular,%
BoldFont       = *-Bold,%
ItalicFont     = *-Italic,%
BoldItalicFont = *-BoldItalic,%
]
\setsansfont{FiraSans}[
Path           = ../../fonts/FiraSans/,%
Extension      = .ttf,%
UprightFont    = *-Regular,%
BoldFont       = *-Bold,%
ItalicFont     = *-Italic,%
BoldItalicFont = *-BoldItalic,%
]
%% MATH FONT: GARAMOND MATH
\setmathfont{FiraMath-Regular}[
  Scale       = MatchLowercase,%
    Extension    = .otf,%
    Path         = ../../fonts/,%
    % StylisticSet = {3, 5},%
]
%% MONO FONT: JULIA MONO
\setmonofont{FiraMono}[
  Scale       = 0.92,%
  Path        = ../../fonts/FiraMono/,%
  Extension   = .ttf,%
  UprightFont = *-Regular,%
  BoldFont    = *-Bold,%
]
"""
}
latex_logo = "media/nmrespy_full.png"
