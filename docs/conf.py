# conf.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 03 May 2022 16:18:55 BST

import os
import sys
import nmrespy
import sphinx_nameko_theme


sys.path.insert(0, os.path.abspath("exts"))
sys.path.insert(0, os.path.abspath(".."))

project = "NMR-EsPy"
copyright = "2022, Simon Hulse & Mohammadali Foroozandeh"
author = "Simon Hulse & Mohammadali Foroozandeh"
version = nmrespy.__version__

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

master_doc = "content/index"

todo_include_todos = True

autosectionlabel_prefix_document = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
\setmainfont{EBGaramond}[
Path           = ../../fonts/ebgaramond/,%
Extension      = .otf,%
UprightFont    = *-Regular,%
BoldFont       = *-Bold,%
ItalicFont     = *-Italic,%
BoldItalicFont = *-BoldItalic,%
]
\setmathfont{Garamond-Math}[
Extension    = .otf,%
Path         = ../../fonts/ebgaramond/,%
StylisticSet = {1,8,5},%
]
\setmonofont{UbuntuMono}[
Path           = ../../fonts/ubuntumono/,%
Extension      = .ttf,%
UprightFont    = *-R,%
BoldFont       = *-B,%
ItalicFont     = *-RI,%
BoldItalicFont = *-BI,%
]
"""
}
