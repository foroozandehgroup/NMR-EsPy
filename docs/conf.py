# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import nmrespy


# -- Project information -----------------------------------------------------

project = 'NMR-EsPy'
copyright = '2021, Simon Hulse'
author = 'Simon Hulse'
version = nmrespy.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.imgmath',
              'sphinx.ext.todo',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
]

imgmath_latex_premable = '\\usepackage{nicefrac}\n'

master_doc = 'index'

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Readthedocs theme
#
# on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
#
# if not on_rtd:  # only import and set the theme if we're building docs locally
#     import sphinx_rtd_theme
#     html_theme = 'sphinx_rtd_theme'
#     html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "bizstyle"

# otherwise, readthedocs.org uses their theme by default, so no need to specify it

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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -----------
# Autosummary
# -----------
autosummary_generate = True
