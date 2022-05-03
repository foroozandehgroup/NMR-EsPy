# setup.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 03 May 2022 16:18:18 BST

from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

exec(open("nmrespy/_version.py", "r").read())

setup(
    name="nmrespy",
    version=__version__,  # noqa: F821
    description="NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python",
    author="Simon G. Hulse, Mohammadali Foroozandeh",
    author_email="simon.hulse@chem.ox.ac.uk",
    url="https://github.com/foroozandehgroup/NMR-EsPy",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "bruker_utils",
        "nmr_sims",
        "colorama; platform_system == 'Windows'",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    packages=find_packages(),
)
