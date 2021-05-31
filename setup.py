from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

exec(open('nmrespy/_version.py').read())

setup(
    name='nmrespy',
    version=__version__,
    description='NMR-EsPy: Nuclear Magnetic Resonance Estimation in Python',
    author='Simon Hulse',
    author_email='simon.hulse@chem.ox.ac.uk',
    url='https://github.com/foroozandehgroup/NMR-EsPy',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
    ],
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "matplotlib>=3.3",
        "colorama==0.4; platform_system == 'Windows'",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    packages=find_packages(),
)
