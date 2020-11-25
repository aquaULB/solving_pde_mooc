---
jupytext:
  formats: ipynb,md:myst
  notebook_metadata_filter: toc
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
toc:
  base_numbering: 1
  nav_menu: {}
  number_sections: true
  sideBar: true
  skip_h1_title: true
  title_cell: Table of Contents
  title_sidebar: Contents
  toc_cell: true
  toc_position: {}
  toc_section_display: true
  toc_window_display: false
---

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>

+++

# Convergence and numba acceleration

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Gauss-Seidel-with-numba" data-toc-modified-id="Gauss-Seidel-with-numba-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Gauss-Seidel with numba</a></span></li></ul></div>

+++

## Introduction

For convenience, we start by importing some modules needed below:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../modules')
# Function to compute an error in L2 norm
from norms import l2_diff

from iter_module import p_exact_2d

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In the previous chapter we have ...

+++

## Gauss-Seidel with numba

```{code-cell} ipython3
from numba import njit
```

```{code-cell} ipython3
# Grid parameters.
nx = 61                  # number of points in the x direction
ny = 61                  # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx - 1)        # grid spacing in the x direction
dy = ly / (ny - 1)        # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing ='ij')

# Compute the rhs
b = (np.sin(np.pi * X / lx) * np.cos(np.pi * Y / ly) +
     np.sin(5.0 * np.pi * X / lx) * np.cos(5.0 * np.pi * Y / ly))
```

```{code-cell} ipython3
def gauss_seidel(p, b, tolerance, max_iter):
    
    nx, ny = b.shape
    iter = 0
    diff = 1
    tol_hist_gs = []
    
    pnew = p.copy()
    
    while (diff > tolerance):
    
        if iter > max_iter:
            print('\nSolution did not converged within the maximum'
                ' number of iterations')
            break
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = ( 0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                                + p[i, j+1] - b[i, j]*dx**2 ))
        
        diff = l2_diff(pnew, p)
        tol_hist_gs.append(diff)

        # Show iteration progress (I would like to add iter but cannot do it)
        # Problems in numba here
        # print('\r', f'diff: {diff:5.2e}', end='')
        
        p = pnew.copy()
        iter += 1

    else:
        print('The solution converged after {:d} iterations'.format(iter))
        return p, tol_hist_gs

@njit
def gauss_seidel_numba(p, b, tolerance, max_iter):
    
    nx, ny = b.shape
    iter = 0
    diff = 1
    tol_hist_gs = []
    
    pnew = p.copy()
    
    while (diff > tolerance):
    
        if iter > max_iter:
            print('\nSolution did not converged within the maximum'
                ' number of iterations')
            break
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                pnew[i, j] = ( 0.25*(pnew[i-1, j] + p[i+1, j] + pnew[i, j-1]
                                + p[i, j+1] - b[i, j]*dx**2 ))
        
        diff = l2_diff(pnew, p)
        tol_hist_gs.append(diff)

        # Show iteration progress (I would like to add iter but cannot do it)
        # Problems in numba here
        # print('\r', f'diff: {diff:5.2e}', end='')
        
        p = pnew.copy()
        iter += 1

    else:
        print('The solution converged after iterations', iter)
        return p, tol_hist_gs
```

```{code-cell} ipython3
p0 = np.zeros((nx,ny))
```

```{code-cell} ipython3
# 20 seconds on my computer (7 times)
%timeit p, tol_hist_gs = gauss_seidel(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
```

```{code-cell} ipython3
# 80ms on my computer (70 times) !!!
%timeit p, tol_hist_gs = gauss_seidel_numba(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
```

```{code-cell} ipython3
# Compute the exact solution (and p for the plot %%timeit does not return it)
p, tol_hist_gs = gauss_seidel_numba(p0.copy(), b, tolerance = 1e-10, max_iter = 1e6)
p_e = p_exact_2d(X, Y)
```

```{code-cell} ipython3
fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(16,5))
# We shall now use the
# matplotlib.pyplot.contourf function.
# As X and Y, we pass the mesh data.
#
# For more info
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contourf.html
#
ax_1.contourf(X, Y, p, 20)
ax_2.contourf(X, Y, p_e, 20)

# plot along the line y=0:
jc = int(ly/(2*dy))
ax_3.plot(x, p_e[:,jc], '*', color='red', markevery=2, label=r'$p_e$')
ax_3.plot(x, p[:,jc], label=r'$pnew$')

# add some labels and titles
ax_1.set_xlabel(r'$x$')
ax_1.set_ylabel(r'$y$')
ax_1.set_title('Exact solution')

ax_2.set_xlabel(r'$x$')
ax_2.set_ylabel(r'$y$')
ax_2.set_title('Numerical solution')

ax_3.set_xlabel(r'$x$')
ax_3.set_ylabel(r'$p$')
ax_3.set_title(r'$p(x,0)$')

ax_3.legend();
```

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```{code-cell} ipython3

```
