# Sphinx fucks up sectioning when it builds the LaTeX docs.
# This file should be run directly after `sphinx-build -b latex . latex`

with open('latex/nmr-espy.tex', 'r') as fh:
    text = fh.read()

    text = text.replace(
        '\\sphinxAtStartPar\nNMR\\sphinxhyphen{}EsPy is available via the',
        '\\chapter{Installation}\n\n\\sphinxAtStartPar\nNMR\\sphinxhyphen{}EsPy is available via the',
    )
    text = text.replace(
        '\\chapter{Python Packages}',
        '\\section{Python Packages}',
    )
    text = text.replace(
        '\\chapter{LaTeX}',
        '\\section{\LaTeX}',
    )
    text = text.replace(
        'follow this link',
        'go to Chapter 4'
    )
    text = text.replace(
        'On this page, \\sphinxcode{\\sphinxupquote{<pyexe>}}',
        'In this section, \\sphinxcode{\\sphinxupquote{<pyexe>}}'
    )
    # text = text.replace(
    #     '\\noindent{\\hspace*{\\fill}\\sphinxincludegraphics[scale=0.7]{{setup_window_plain}.png}\\hspace*{\\fill}}',
    #     ''
    # )
    text = text.replace(
        '\\chapter{Contributors}\n'
        '\\label{\\detokenize{contributors:contributors}}'
        '\\label{\\detokenize{contributors::doc}}',
        '\\chapter{Contributors}\n'
        '\\label{\\detokenize{contributors:contributors}}'
        '\\label{\\detokenize{contributors::doc}}\n\n'
        '\\begin{center}\n\\begin{tabular}{ m{2.2in} m{2.2in} }\n'
        '\\includegraphics[width=2in]{../media/contributors/SH.jpg} & '
        '\\textbf{Simon Hulse} \\newline \\textit{Ph.D. Student 2019-} '
        '\\newline Project\'s main developer \\\\\n'
        '\\includegraphics[width=2in]{../media/contributors/MF.jpg} & '
        '\\textbf{Mohammadali Foroozandeh} \\newline '
        '\\textit{Simon\'s Ph.D. supervisor} \\\\\n'
        '\\includegraphics[width=2in]{../media/contributors/TM.png} & '
        '\\textbf{Thomas Moss} \\newline \\textit{MChem Student '
        'Jan 2020 - June 2020} \\newline Helped in extending '
        'NMR-EsPy to support 2D data \\\\\n'
        '\\end{tabular}\n'
        '\\end{center}'
    )

with open('latex/nmr-espy.tex', 'w') as fh:
    fh.write(text)
