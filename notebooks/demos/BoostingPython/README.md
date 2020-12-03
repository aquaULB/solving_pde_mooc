# Boosting Python

This little project provides a demo of how to speed up the Python code using [Numba][1] or [Cython][2]. As a model problem, solution of two-dimensional Poisson equation using Gauss-Seidel method is provided.

## Structure

The project has the following structure:
* *cy*
    * csolver.pyx
    * setup.py
    * *lib*
* *py*
    * solvers.py
* README.md

*cy* directory contains source Cython code (csolver.pyx), the makefile (setup.py) and libraries compiled from the source code on different platforms (in *lib*).

*py* directory contains Python module solvers.py that contains a Gauss-Seidel solver that can be run with or without using Numba.

## Using Cython solver

In order to use Cython solver, import the library (assuming the path to *cy* has been added to the PYTHONPATH):
```
import cy.lib.csolver as csolver
```

If you wish to recompile the source code on your platform, you must have Cython and C compilers installed. We refer you to the [Cython docs][3] for installation instructions. Note that you can also install Cython with conda:
```
conda install -c conda-forge cython
```
C compiler though, if not present, must be installed separately.

To execute compilation instructions, run from *cy* directory:
```
python setup.py build_ext --build-lib <target_dir>
```
After compilation is complete, library will be located in *<target_dir>*.

Run Gauss-Seidel solver from the Python script after it has been imported:
```
<...>, <...>, <...> = csolver.c_gauss_seidel(c_p, b, dx, tol, max_it)
```
Meaning of function parameters and returns is explained in documentation of function.

To display documentation from IPython:
```
%pinfo csolver.c_gauss_seidel
```

## Using Python solver

In order to use Python solver, import the module:
```
import py.solvers as solver
```
Access Gauss-Seidel solver in the following way:
```
<...>, <...>, <...> = solver.gauss_seidel(nb_p, b, dx, tol, max_it, use_numba=<...>)
```
Note that `use_numba` flag of `solver.gauss_seidel` is set to `False` by default.

Meaning of function parameters and returns is explained in documentation of function.

To display documentation from IPython:
```
%pinfo solver.gauss_seidel
```

[1]: <https://numba.pydata.org> "Numba"
[2]: <https://cython.readthedocs.io/en/latest/#> "Cython"
[3]: <https://cython.readthedocs.io/en/latest/src/quickstart/install.html> "Cython installation"