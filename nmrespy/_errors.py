#!/usr/bin/python3

# nmrespy._errors
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# 

from ._cols import *
if USE_COLORAMA:
    import colorama


class MoreThanTwoDimError(Exception):
    """Raise when user tries importing data that is >2D"""

    def __init__(self):
        self.msg = f'{R}nmrespy does\'nt support >2D data.{END}'
        super().__init__(self.msg)


class TwoDimUnsupportedError(Exception):
    """Raise when user tries running a method that doesn't support 2D
    data yet"""

    def __init__(self):
        self.msg = f'{R}Unfortunately 2D virtual echo creation isn\'t' \
                   + ' supported yet. Check if there are any more recent' \
                   + ' versions of nmrespy with this feature:\n' \
                   + f'{C}http://foroozandeh.chem.ox.ac.uk/home{END}'
        super().__init__(self.msg)


class InvalidUnitError(Exception):
    """Raise when the specified unit is invalid"""

    def __init__(self, *args):
        self.msg = f'{R}unit should be one of the following: '
        self.msg += ', '.join([repr(unit) for unit in args]) + END
        super().__init__(self.msg)


class InvalidDirectoryError(Exception):
    """Raise when the a dictionary does not have the requisite files"""

    def __init__(self, dir):
        self.msg = f'{R}{dir} does not contain the necessary files' \
                   + f' for importing data.{END}'
        super().__init__(self.msg)


class NoParameterEstimateError(Exception):
    """Raise when the a dictionary does not have the requisite files"""

    def __init__(self):
        self.msg = f'{R}No attribute corresponding to a parameter array' \
                   + f' could be found. Perhaps you need to run an' \
                   + f' estimation routine first?{END}'
        super().__init__(self.msg)


class NoSuitableDataError(Exception):
    """Raise when user tries to run mpm/nlp on a class that has imported
    pdata, and not constructed a virtual echo from it"""

    def __init__(self):
        self.msg = f'{R}No appropriate data to analyse was found.' \
                   + f' It is possible that this is because you have' \
                   + f' imported processed data, and have not yet' \
                   + f' generated a virtual echo from it.{END}'
        super().__init__(self.msg)


class PhaseVarianceAmbiguityError(Exception):
    """Raise when phase_variance is True, but 'p' is not specified in mode"""

    def __init__(self, mode):
        self.msg = f'{R}You have specified you want to minimise phase' \
                   + f' varaince (phase_variance=True) but you have not' \
                   + f' asked for the phases to be be optimised' \
                   + f' (mode = \'{mode}\'). The phase variance cannot change' \
                   + f' if you don\'t include \'p\' in mode.{END}'
        super().__init__(self.msg)


class AttributeIsNoneError(Exception):
    """Raise when the user calls a ``get_<attr>`` method, but the attribute
    is None"""

    def __init__(self, attribute, method):
        self.msg = f'{R}The attribute {attribute} is None. Perhaps you are' \
                   + f' yet to call {method} on the class instance?{END}'
        super().__init__(self.msg)


class LaTeXFailedError(Exception):
    """Raise when the user calls write_result(), with format set to 'pdf',
    but compiling the TeX file failed when pdflatex was called."""

    def __init__(self, texpath):
        self.msg = f'{R}The file {texpath} failed to compile using' \
                   + f' pdflatex. Make you sure have a LaTeX installation' \
                   + f' by opening a terminal and entering:\n{C}pdflatex\n' \
                   + f'{R}If you have pdflatex, run:\n{C}pdflatex {texpath}\n' \
                   + f'{R}and try to fix whatever it is not happy with{END}'
        super().__init__(self.msg)
