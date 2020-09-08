#!/usr/bin/python3
# nmrespy.errors
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# Custom errors

from ._cols import *
if USE_COLORAMA is True:
    import colorama

class MoreThanTwoDimError(Exception):
    """Raise when user tries importing data that is >2D"""
    def __init__(self):
        self.msg = f'\n{R}nmrespy does\'nt support >2D data.{END}'
        super().__init__(self.msg)


class TwoDimUnsupportedError(Exception):
    """Raise when user tries running a method that doesn't support 2D
    data yet"""
    def __init__(self):
        self.msg = f'\n{R}Unfortunately 2D virtual echo creation isn\'t' \
                   + ' supported yet. Check if there are any more recent' \
                   + ' versions of nmrespy with this feature:\n' \
                   + f'{C}http://foroozandeh.chem.ox.ac.uk/home{END}'
        super().__init__(self.msg)


class InvalidUnitError(Exception):
    """Raise when the specified unit is invalid"""
    def __init__(self, *args):
        self.msg = f'\n{R}unit should be one of the following: '
        for unit in args:
            self.msg += f'\'{unit}\', '
        self.msg = self.msg[:-2] + f'{END}'
        super().__init__(self.msg)


class InvalidDirectoryError(Exception):
    """Raise when the a dictionary does not have the requisite files"""
    def __init__(self, dir):
        self.msg = f'\n{R}{dir} does not contain the necessary files' \
                   + f' for importing data.{END}'
        super().__init__(self.msg)

class NoParameterEstimateError(Exception):
    """Raise when the a dictionary does not have the requisite files"""
    def __init__(self):
        self.msg = f'\n{R}No attribute corresponding to a parameter array' \
                   + f' could be found. Perhaps you need to run an' \
                   + f' estimation routine first?{END}'
        super().__init__(self.msg)

class NoSuitableDataError(Exception):
    """Raise when user tries to run mpm/nlp on a class that has imported
    pdata, and no constructed virtual echo from it"""
    def __init__(self):
        self.msg = f'\n{R}No appropriate data to analyse was found.' \
                   + f' It is possible that this is because you have' \
                   + f' imported processed data, and have not yet' \
                   + f' generated a virtual echo from it.{END}'
        super().__init__(self.msg)


class PhaseVarianceAmbiguityError(Exception):
    """Raise when phase_variance is True, but 'p' is not specified in mode"""
    def __init__(self, mode):
        self.msg = f'\n{R}You have specified you want to minimise phase' \
                   + f' varaince (phase_variance=True) but you have not' \
                   + f' asked for the phases to be be optimised' \
                   + f' (mode = \'{mode}\'). The phase variance cannot change' \
                   + f' if you don\'t include \'p\' in mode.{END}'
        super().__init__(self.msg)


class AttributeNotFoundError(Exception):
    """Raise when the user calls a get_<attr>() method, but the attribute
    doesn't exists, as it is necessary to generate it using other methods
    first"""
    def __init__(self, attribute, method):
        self.msg = f'\n{R}{attribute} doesn\'t exist. This is' \
                   + f' generated after calling {method} on the class.' \
                   + f' Perhaps you are yet to do this?{END}'
        super().__init__(self.msg)

class LaTeXFailedError(Exception):
    """Raise when the user calls a get method, but the attribute doesn't
    exists, as it is necessary to generate it using other methods first"""
    def __init__(self, texpath):
        self.msg = f'\n{R}The file {texpath} failed to compile using' \
                   + f' pdflatex. Make you sure have a LaTeX installation' \
                   + f' by opening a terminal and entering:\n{C}pdflatex\n' \
                   + f'{R}If you have pdflatex, run:\n{C}pdflatex {texpath}\n' \
                   + f'{R}and try to fix whatever it is not happy with{END}'
        super().__init__(self.msg)
