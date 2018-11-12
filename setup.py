#!/usr/bin/env python
# -*- coding: utf-8 -*-


import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'delayedsparse'
DESCRIPTION = 'Delayed sparse matrix in Python'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/niitsuma/delayedsparse'
EMAIL = 'hirotaka.niitsuma@gmail.com'
AUTHOR = 'Hirotaka Niitsuma'
REQUIRES_PYTHON = '>=3.4.0'
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy>=1.14.0',
    'scipy>=1.0.1',
    'scikit-learn>=0.19.0'
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    license='You can use these codes olny for self evaluation. Cannot use these codes for commercial and academical use.',
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('examples','tests',)),
    package_data = {'': ['LICENSE']},
    install_requires=REQUIRED,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',    
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    
)
