# latex_fudge.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 09 Jan 2023 13:39:00 GMT

# This file should be run directly after `sphinx-build -b latex . latex`
# You should be in the `docs` drectory when this is run.
from pathlib import Path
import re
import socket

hostname = socket.gethostname()
if hostname in ("precision", "spectre"):
    basedir = Path("~/Documents/DPhil").expanduser()
elif hostname in ("belladonna.chem.ox.ac.uk", "parsley.chem.ox.ac.uk"):
    basedir = Path("~/DPhil").expanduser()
fname = basedir / "projects/spectral_estimation/NMR-EsPy/docs/_build/latex/nmr-espy.tex"

with open(fname, 'r') as fh:
    text = fh.read()

text = re.sub(
    r' LaTeX\.',
    r' \\LaTeX.',
    text
)
text = re.sub(
    r' LaTeX(\s?)',
    r' \\LaTeX\ \1',
    text
)
text = text.replace(
    'follow this link',
    'go to Chapter 4'
)
text = text.replace(
    'On this page, \\sphinxcode{\\sphinxupquote{<pyexe>}}',
    'In this section, \\sphinxcode{\\sphinxupquote{<pyexe>}}'
)

# Create nicely formatted tabular of contributors
text = text.replace(
    '\\chapter{Contributors}\n'
    '\\label{\\detokenize{contributors:contributors}}'
    '\\label{\\detokenize{contributors::doc}}',
    '\\chapter{Contributors}\n'
    '\\label{\\detokenize{contributors:contributors}}'
    '\\label{\\detokenize{contributors::doc}}\n\n'
    '\\begin{center}\n\\begin{tabular}{ m{2.2in} m{2.2in} }\n'
    '\\includegraphics[width=2in]{../../media/contributors/SH.jpg} & '
    '\\textbf{Simon Hulse} \\newline \\textit{Ph.D. Student 2019-2023} '
    '\\newline Project\'s main developer \\\\\n'
    '\\includegraphics[width=2in]{../../media/contributors/MF.jpg} & '
    '\\textbf{Mohammadali Foroozandeh} \\newline '
    '\\textit{Simon\'s Ph.D. supervisor 2019-2022} \\newline '
    'Now working at Zurich Instruments\\\\\n'
    '\\includegraphics[width=2in]{../../media/contributors/TM.png} & '
    '\\textbf{Thomas Moss} \\newline \\textit{MChem Student '
    'Sep 2020 - June 2021} \\newline Helped in extending '
    'NMR-EsPy to support 2D data \\newline Now studying graduate-entry medicine\\\\\n'
    '\\end{tabular}\n'
    '\\end{center}'
)

# Get rid of Python Module Index at end
text = re.sub(
    r'\\renewcommand\{\\indexname\}\{Python\sModule\sIndex\}(.*)\\printindex',
    '',
    text,
    flags=re.DOTALL
)

with open(fname, 'w') as fh:
    fh.write(text)
