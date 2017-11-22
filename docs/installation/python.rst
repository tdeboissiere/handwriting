
Python configuration
=====================

The environment to reproduce the results can be obtained by running:

.. code::

    $ bash setup_anaconda.sh


- This downloads miniconda3, saves it to the `$HOME` folder and installs all the libraries needed in a virtual environment.
- It detects a CUDA installation and installs the cuda version of pytorch if CUDA is detected.

To activate the virtual environment:

.. code::

    $ bash virtualenv.sh