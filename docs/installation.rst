************
Installation
************

Requirements
============

ODHIN has the following strict requirements:

- Python version 2.7 or 3.3+
- Numpy version 1.8 or above
- Scipy version 0.14 or above
- Matplotlib version 1.4 or above
- Astropy version 1.0 or above


Installing with pip
===================

To install the development version, first clone the git repository::

    git clone https://git-cral.univ-lyon1.fr/raphael.bacher/deblend

If you want to use conda, a preconfigured environment is available::

    conda env create -f environment.yml
    conda activate odhin3

Then, you can install the package::

    pip install .
