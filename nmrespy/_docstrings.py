# _docstrings.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 18 Jan 2023 14:40:02 GMT

def fixdocstring(func):
    func.__doc__ = func.__doc__.replace(
        "<index_description>",
        "The index of the result to edit. Index ``0`` corresponds to the "
        "first result obtained using the estimator, ``1`` corresponds to the "
        "next, etc. By default, the most recently obtained result will be "
        "edited."
    )
