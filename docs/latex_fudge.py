# Sphinx fucks up sectioning in some places when it builds the LaTeX docs.
# Also, idk how to implement certain things in LaTeX via rst files,
# so this file takes the .tex file output by sphinx and manually
# adjusts it to make any desired changes.

# This file should be run directly after `sphinx-build -b latex . latex`
# You should be in the `docs` drectory when this is run.
import re
import socket

hostname = socket.gethostname()
if hostname == 'precision':
    fname = ('/home/simon/Documents/DPhil/projects/spectral_estimation/'
             'NMR-EsPy/docs/build/latex/nmr-espy.tex')
elif hostname == 'belladonna.chem.ox.ac.uk':
    fname = ('/u/mf/jesu2901/Documents/DPhil/projects/spectral_estimation/'
             'NMR-EsPy/docs/build/latex/nmr-espy.tex')

with open(fname, 'r') as fh:
    text = fh.read()

# Fix messed up titling with installation chapter
text = text.replace(
    '\\sphinxAtStartPar\nNMR\\sphinxhyphen{}EsPy is available via the',
    '\\chapter{Installation}\n\n\\sphinxAtStartPar\nNMR\\sphinxhyphen{}EsPy '
    'is available via the',
)
text = text.replace(
    '\\chapter{Python Packages}',
    '\\section{Python Packages}',
)
text = text.replace(
    '\\chapter{LaTeX}',
    '\\section{LaTeX}',
)
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
# Replace "follow this link" with "go to Chapter 4"
text = text.replace(
    'follow this link',
    'go to Chapter 4'
)
# Replace "On this page" with "In this section"
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
    '\\textbf{Simon Hulse} \\newline \\textit{Ph.D. Student 2019-} '
    '\\newline Project\'s main developer \\\\\n'
    '\\includegraphics[width=2in]{../../media/contributors/MF.jpg} & '
    '\\textbf{Mohammadali Foroozandeh} \\newline '
    '\\textit{Simon\'s Ph.D. supervisor} \\\\\n'
    '\\includegraphics[width=2in]{../../media/contributors/TM.png} & '
    '\\textbf{Thomas Moss} \\newline \\textit{MChem Student '
    'Jan 2020 - June 2020} \\newline Helped in extending '
    'NMR-EsPy to support 2D data \\\\\n'
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
