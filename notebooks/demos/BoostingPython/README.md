# Boosting Python

This little project provides a demo of how to speed up the Python code using [Numba][1] or [Cython][2] by solving two-dimensional Poisson equation using Gauss-Seidel method.

The project has the following structure:
* *cy*
    * csolver.pyx
    * setup.py
    * *lib*
    * *build*
* *py*
    * solvers.py
* README.md

*cy* directory contains source Cython code (csolver.pyx), the makefile (setup.py), library compiled from the source code (in *lib*) and the files generated at the compile time (in *build*).

In order to use Cython solver, import the library (assuming the path to *cy* has been added to the PYTHONPATH):
```
import cy.lib.csolver as csolver
```

In order to recompile on other platform, run from *cy* directory:
```
python setup.py build_ext --build-lib <target_dir>
```
Library will be compiled in *<target_dir>*.

Access Gauss-Seidel solver in the following way:
```
csolver.c_gauss_seidel(c_p, b, dx, tol, max_it)
```

*py* directory contains Python module solvers.py that contains a Gauss-Seidel solver that can be run with or without using Numba.

In order to use Python solver, import the module:
```
import py.solvers as solver
```
Access Gauss-Seidel solver in the following way:
```
solver.gauss_seidel(nb_p, b, dx, tol, max_it, use_numba=True)
```

Meaning of parameters and returns is accessible in the documentation of the function. In IPython run:
```
%pinfo <name_of_function>
```

Note that `use_numba` flag of `solver.gauss_seidel` is set to `False` by default.

[1]: <https://numba.pydata.org> "Numba"
[2]: <https://cython.readthedocs.io/en/latest/#> "Cython"