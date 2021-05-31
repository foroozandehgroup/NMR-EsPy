# _errors
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""nmrespy-specific errors"""

from nmrespy import *
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()


class MoreThanTwoDimError(Exception):
    """Raise when user tries importing data that is >2D"""

    def __init__(self):
        self.msg = f'{cols.R}nmrespy does\'nt support >2D data.{cols.END}'
        super().__init__(self.msg)


class TwoDimUnsupportedError(Exception):
    """Raise when user tries running a method that doesn't support 2D
    data yet"""

    def __init__(self):
        self.msg = (f'{cols.R}Unfortunately 2D virtual echo creation isn\'t'
                    ' supported yet. Check if there are any more recent'
                    ' versions of nmrespy with this feature:\n'
                    f'{cols.C}{GITHUBLINK}{cols.END}')
        super().__init__(self.msg)


class InvalidUnitError(Exception):
    """Raise when the specified unit is invalid"""

    def __init__(self, *args):
        valid_units = ', '.join([repr(unit) for unit in args])
        self.msg = (f'{cols.R}unit should be one of the following:'
                    f' {valid_units}{cols.END}')
        super().__init__(self.msg)


class InvalidDirectoryError(Exception):
    """Raise when the a dictionary does not have the requisite files"""

    def __init__(self, dir):
        self.msg = (f'{cols.R}{str(dir)} does not contain the necessary files'
                    f' for importing data.{cols.END}')
        super().__init__(self.msg)


class ParameterNotFoundError(Exception):
    """Raise when a desired parameter is not present in an acqus/procs file"""

    def __init__(self, param_name, path):
        self.msg = (f'{cols.R}Could not find parameter {param_name} in file'
                    f'{str(path)}{cols.END}')
        super().__init__(self.msg)


class NoParameterEstimateError(Exception):
    """Raise when instance does not possess a valid parameter array"""

    def __init__(self):
        self.msg = (f'{cols.R}No attribute corresponding to a parameter array'
                    f' could be found. Perhaps you need to run an'
                    f' estimation routine first?{cols.END}')
        super().__init__(self.msg)


class PhaseVarianceAmbiguityError(Exception):
    """Raise when phase_variance is True, but 'p' is not specified in mode"""

    def __init__(self, mode):
        self.msg = (f'{cols.R}You have specified you want to minimise phase'
                    ' varaince (phase_variance=True) but you have not'
                    ' asked for the phases to be be optimised'
                    f' (mode = \'{mode}\'). The phase variance cannot change'
                    f' if you don\'t include \'p\' in mode.{cols.END}')
        super().__init__(self.msg)


class AttributeIsNoneError(Exception):
    """Raise when the user calls a `get_<attr>` method, but the attribute
    is None"""

    def __init__(self, attribute, method):
        self.msg = f'{cols.R}The attribute {attribute} is None.'
        if method is not None:
            self.msg += (
                f' Perhaps you are yet to call {method} on the class'
                f' instance?'
            )

        self.msg += cols.END
        super().__init__(self.msg)


class LaTeXFailedError(Exception):
    """Raise when the user calls write_result(), with format set to 'pdf',
    but compiling the TeX file failed when pdflatex was called."""

    def __init__(self, texpath):
        self.msg = (f'{cols.R}The file {texpath} failed to compile using'
                    ' pdflatex. Make you sure have a LaTeX installation'
                    f' by opening a terminal and entering:\n{cols.C}'
                    f' pdflatex\n{cols.R}If you have pdflatex, run:\n'
                    f' {cols.C}pdflatex {texpath}\n{cols.R}and try to fix'
                    f' whatever it is not happy with{cols.END}')

        super().__init__(self.msg)
