# DaetPy
Python code for processing, analysing, and plotting DAET experiments. Optimised for data acquired with PLACE (see [PALab/place](https://github.com/PALab/place)).

# About DaetPy

The modular design of the code, centred around the DaetPy object, provides a convenient way to construct a workflow from simple scripts which perform individual tasks. These simple "control scripts" are short files which first initialise the DaetPy object, and then call one or more object methods in a few lines to process or display the data.

# Installation

daetpy is installed as a Python module, so that it can be imported like any other library. Once downloaded, run the setup.py script in the DaetPy directory by typing in a terminal:

```
$ python setup.py develop
```

After this, you should be able to import daetpy into a Python script using:

```
from daetpy.main import DaetPy
```

A simple plotting example is provided, along with example data. The code has extensive documentation for all functions within main.py and the auxilliary scripts.

# Contact
__Author:__ Jonathan Simpson, jsimpsonUoA

__Email:__ jsim921@aucklanduni.ac.nz

2020, Physical Acoustics Lab,
The Universtiy of Auckland, New Zealand
