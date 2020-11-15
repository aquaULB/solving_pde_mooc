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

# One dimensional heat equation: implicit methods

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Backward-Euler-method" data-toc-modified-id="Backward-Euler-method-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Backward Euler method</a></span></li></ul></div>

+++

## Introduction

+++

For convenience, we start by importing some modules needed below:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In the previous notebook we have described some explicit methods to solve the one dimensional heat equation;

\begin{equation}
\label{eq:1Dheat2}
   \partial_t T(x,t) = \alpha \frac{d^2 T} {dx^2}(x,t) + \sigma (x,t).
\end{equation}

where $T$ is the temperature and $\sigma$ is an optional heat source term.

In all cases considered, we have observed that stability of the algorithm requires a restriction on the time proportional to $\Delta x^2$. As this is rather restrictive, we focus here on some implicit methods and see how they compare.

+++

## Backward Euler method

+++

We begin by considering the backward Euler time advancement scheme in combination with the second-order accurate centered finite difference formula for $d^2T/dx^2$ and we do not include the source term for the stability analysis.

We recall that for a generic ordinary differential equation $y'=f(y,t)$, the backward Euler method is,

\begin{equation}
y^{n+1} = y^{n} + \Delta t f(y^{n+1},t).
\end{equation}

In our case, the discretization is therefore,

\begin{align}
\label{heatbackwardForward}
& \frac{T^{n+1}_i - T^n_i}{\Delta t}=\alpha \frac{T^{n+1}_{i-1}-2T^{n+1}_i+T^{n+1}_{i+1}}{\Delta x^2}
\end{align}

In matrix notation this is equivalent to:

\begin{equation}
\label{eq:matHeatBackwardEuler}
    (I-A)\boldsymbol{T}^{n+1} = \boldsymbol{T}^{n},
\end{equation}

where $I$ is the identity matrix. If we adopt Dirichlet boundary conditions, the matrix $A$ is identical to its counterpart for the forward Euler scheme (see previous notebook). For reference, the eigenvalues $m_k$ of $A$ are real and negative:

\begin{equation}
m_k = 2\frac{\alpha dt}{dx^2}\left(\cos\left(\frac{\pi k}{nx}\right)-1\right),\; k=1,\ldots, nx-2.
\end{equation}

If we diagonalize $A$ and denote by $\boldsymbol z=(z_1,\ldots,z_{nx-2})$ the coordinates of $\boldsymbol T$ in the basis of eigenvectors, we have:

\begin{equation}
(I-\Lambda) \boldsymbol z^{n+1} = \boldsymbol z^n
\end{equation}

where $\Lambda$ is the diagonal matrix containing the eigenvalues of $A$. Each of the $nx-2$ decoupled equations may therefore be written as:

\begin{equation}
z_k^n = \left(\frac{1}{1-m_k}\right)^n z_k^0 
\end{equation}

Since $m_k < 0\; \forall k$, all the coordinates $z_k$ remain bounded and the algorithm is unconditionally stable.

We can also examine the stability of the algorithm for the case of periodic boundary conditions using the modified wavenumber analysis. We adopt the same notations as in the previous notebook and get the same expression for the evolution of the Fourier modes,

\begin{align}
\label{eq:matHeatBackwardEulerFourier}
    \frac{d\hat T(k_m,t)}{dt}&=\alpha\left(\frac{e^{-ik_m \Delta x} - 2 + e^{ik_m \Delta x}}{\Delta x^2}\right) \hat{T}(k_m,t) \\
    &= -\alpha k_m'^2 \hat T(k_m,t)
\end{align}

with the modified wavenumbers being real and defined by,

\begin{equation}
k_m'^2 = \frac{2}{\Delta x^2}(1 - \cos(k_m \Delta x)).
\end{equation}

Because all the coefficients $-\alpha k_m'^2$ are real and negative, the Fourier components remain bounded if we discretize eq. \ref{eq:matHeatBackwardEulerFourier} using the backward time integration method. The algorithm is therefore also unconditionally stable in the case of periodic boundary conditions.

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
