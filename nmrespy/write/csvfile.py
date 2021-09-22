def _write_csv(
    path, description, info_headings, info, param_titles, param_table,
    fprint,
):
    """Writes parameter estimate to a CSV.

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

    with open(path, 'w', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        # Timestamp
        writer.writerow([_timestamp().replace('\n', ' ')])
        writer.writerow([])
        # Description
        if description is not None:
            writer.writerow(['Description:', description])
            writer.writerow([])
        # Experiment info
        if info is not None:
            writer.writerow(['Experiment Info:'])
            for row in zip(info_headings, info):
                writer.writerow(row)
            writer.writerow([])
        # Parameter table
        writer.writerow(['Result:'])
        writer.writerow(param_titles)
        for row in param_table:
            writer.writerow(row)

    if fprint:
        print(f'{GRE}Saved result to {path}{END}')


