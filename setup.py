import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()

exec(open('nmrespy/_version.py').read())

setup(
    name='nmrespy',
    version=__version__,
    description='NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python',
    author='Simon Hulse',
    author_email='simon.hulse@chem.ox.ac.uk',
    url='https://github.com/5hulse/NMR-EsPy',
    long_description=long_description,
    long_description_content_type="text/restructuredtext",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
