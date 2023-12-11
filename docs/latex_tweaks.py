# latex_tweaks.py
# Simon Hulse
# simonhulse@protonmail.com
# Last Edited: Mon 11 Dec 2023 14:42:20 EST

from pathlib import Path


texfile = Path("./_build/latex/nmr-espy.tex").resolve()
assert texfile.is_file()

with open(texfile, "r") as fh:
    lines = fh.readlines()

nlines = len(lines)
for i, line in enumerate(reversed(lines)):
    if "Python Module Index" in line:
        index = nlines - 1 - i
        break

lines = lines[:index - 1] + [lines[-1]]
with open(texfile, "w") as fh:
    fh.writelines(lines)
