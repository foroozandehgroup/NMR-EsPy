def _write_pdf(
    path, description, info_headings, info, param_titles, param_table,
    pdflatex_exe, fprint,
):
    """Writes parameter estimate to a PDF using ``pdflatex``.

    Parameters
    -----------
    path : pathlib.Path
        File path

    description : str or None, default: None
        A descriptive statement.

    info_headings : list or None, default: None
        Headings for experiment information.

    info : list or None, default: None
        Information that corresponds to each heading in `info_headings`.

    param_titles : list
        Titles for parameter array table.

    param_table : list
        Array of contents to append to the result table.

    fprint: bool
        Specifies whether or not to print output to terminal.
    """
    # Open text of template .tex file which will be amended
    with open(NMRESPYPATH / 'config/latex_template.txt', 'r') as fh:
        txt = fh.read()

    # Add image paths and weblinks to TeX document
    # If on Windows, have to replace paths of the form:
    # C:\a\b\c
    # to:
    # C:/a/b/c
    patterns = (
        '<MFLOGOPATH>',
        '<NMRESPYLOGOPATH>',
        '<DOCSLINK>',
        '<MFGROUPLINK>',
        '<BOOKICONPATH>',
        '<GITHUBLINK>',
        '<GITHUBLOGOPATH>',
        '<MAILTOLINK>',
        '<EMAILICONPATH>',
        '<TIMESTAMP>'
    )

    paths = (
        MFLOGOPATH,
        NMRESPYLOGOPATH,
        DOCSLINK,
        MFGROUPLINK,
        BOOKICONPATH,
        GITHUBLINK,
        GITHUBLOGOPATH,
        MAILTOLINK,
        EMAILICONPATH,
    )

    for pattern, path_ in zip(patterns, paths):
        txt = txt.replace(pattern, str(path_).replace('\\', '/'))

    # Include a timestamp
    txt = txt.replace('<TIMESTAMP>', _timestamp().replace('\n', '\\\\'))

    # --- Description ----------------------------------------------------
    if description is None:
        # No description given, remove relavent section of .tex file
        txt = txt.replace(
            '% user provided description\n\\subsection*{Description}\n'
            '<DESCRIPTION>',
            '',
        )

    else:
        txt = txt.replace('<DESCRIPTION>', description)

    # --- Experiment Info ------------------------------------------------
    if info is None:
        # No info given, remove relavent section of .tex file
        txt = txt.replace(
            '\n% experiment parameters\n'
            '\\subsection*{Experiment Information}\n'
            '\\hspace{-6pt}\n'
            '\\begin{tabular}{ll}\n<INFOTABLE>\n'
            '\\end{tabular}\n',
            '',
        )

    else:
        # Construct 2-column tabular of experiment info headings and values
        rows = list(list(row) for row in zip(info_headings, info))
        info_table = _latex_tabular(rows)
        txt = txt.replace('<INFOTABLE>', info_table)

    # --- Parameter Table ------------------------------------------------
    # Determine number of columns required
    txt = txt.replace('<COLUMNS>', len(param_titles) * 'c')
    # Construct parameter title and table body
    txt = txt.replace('<PARAMTITLES>', _latex_tabular([param_titles]))
    txt = txt.replace('<PARAMTABLE>', _latex_tabular(param_table))

    # Incude plus-minus symbol. For denoting errors.
    txt = txt.replace("±", "$\\pm$ ")

    # TODO support for including result figure
    txt = txt.replace(
        '% figure of result\n\\begin{center}\n'
        '\\includegraphics{<FIGURE_PATH>}\n\\end{center}\n',
        '',
    )

    # TODO
    # Put all LatEx compilation stuff in separate function
    # compile_status = _compile_latex_pdf()

    # --- Generate PDF using pdflatex ------------------------------------
    # Create required file paths:
    # .tex and .pdf paths with temporary directory (this is where the files
    # will be initially created)
    # .tex and .pdf files with desired directory (files will be moved from
    # temporary directory to desired directory once pdflatex is run).
    tex_tmp_path = Path(tempfile.gettempdir()) / path.with_suffix('.tex').name
    pdf_tmp_path = Path(tempfile.gettempdir()) / path.name
    tex_final_path = path.with_suffix('.tex')
    pdf_final_path = path

    # Write contents to cwd tex file
    with open(tex_tmp_path, 'w', encoding='utf-8') as fh:
        fh.write(txt)

    try:
        if pdflatex_exe is None:
            pdflatex_exe = "pdflatex"
        # -halt-on-error flag is vital. If any error arises in running
        # pdflatex, the program would get stuck
        subprocess.run(
            [pdflatex_exe,
             '-halt-on-error',
             f'-output-directory={tex_tmp_path.parent}',
             tex_tmp_path],
            stdout=subprocess.DEVNULL,
            check=True,
        )

        # Move pdf and tex files from temp directory to desired directory
        shutil.move(tex_tmp_path, tex_final_path)
        shutil.move(pdf_tmp_path, pdf_final_path)

    except subprocess.CalledProcessError:
        # pdflatex came across an error
        shutil.move(tex_tmp_path, tex_final_path)
        raise _errors.LaTeXFailedError(tex_final_path)

    except FileNotFoundError:
        # Most probably, pdflatex does not exist
        raise _errors.LaTeXFailedError(tex_final_path)

    # Remove other LaTeX files
    os.remove(tex_tmp_path.with_suffix('.out'))
    os.remove(tex_tmp_path.with_suffix('.aux'))
    os.remove(tex_tmp_path.with_suffix('.log'))

    # # TODO: remove figure file if it exists
    # try:
    #     os.remove(figure_path)
    # except UnboundLocalError:
    #     pass

    # Print success message
    if fprint:
        print(f'{GRE}Result successfuly output to:\n'
              f'{pdf_final_path}\n'
              'If you wish to customise the document, the TeX file can'
              ' be found at:\n'
              f'{tex_final_path}{END}')


def _latex_tabular(rows):
    """Creates a string of text that denotes a tabular entity in LaTeX

    Parameters
    ----------
    rows : list
        Nested list, with each sublist containing elements of a single row
        of the table.

    Returns
    -------
    table : str
        LaTeX-formated table

    Example
    -------
    .. code:: python3

       >>> from nmrespy.write import _latex_tabular
       >>> rows = [['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']]
       >>> print(_latex_tabular(rows))
       A1 & A2 & A3 \\\\
       B1 & B2 & B3 \\\\
    """
    table = ''
    for row in rows:
        table += ' & '.join([e for e in row]) + ' \\\\\n'
    return table
