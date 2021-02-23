import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()

exec(open('nmrespy/_version.py').read())

setuptools.setup(
    name='nmrespy_testing0',
    version=__version__,
    description='NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python',
    author='Simon Hulse',
    author_email='simon.hulse@chem.ox.ac.uk',
    url='https://github.com/foroozandehgroup/NMR-EsPy',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.20"
        "scipy>=1.6"
        "matplotlib>=3.3"
        "colorama==0.4 ; platform_system == 'Windows'"
    ],
    python_requires='>=3.7',
)
