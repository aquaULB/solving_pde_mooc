---
jupytext:
  formats: ipynb,md:myst
  notebook_metadata_filter: toc
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
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
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Backward-Euler-method" data-toc-modified-id="Backward-Euler-method-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Backward Euler method</a></span></li><li><span><a href="#Crank-Nicolson-method" data-toc-modified-id="Crank-Nicolson-method-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Crank-Nicolson method</a></span></li><li><span><a href="#Numerical-solution" data-toc-modified-id="Numerical-solution-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Numerical solution</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>

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

In all cases considered, we have observed that stability of the algorithm requires a restriction on the time step proportional to $\Delta x^2$. As this is rather restrictive, we focus here on some implicit methods and see how they compare.

+++

## Backward Euler method

+++

We begin by considering the backward Euler time advancement scheme in combination with the second-order accurate centered finite difference formula for $d^2T/dx^2$ and we do not include the source term for the stability analysis.

We recall that for a generic ordinary differential equation $y'=f(y,t)$, the backward Euler method is,

\begin{equation*}
y^{n+1} = y^{n} + \Delta t f(y^{n+1},t).
\end{equation*}

In our case, the discretization is therefore,

\begin{align*}
\frac{T^{n+1}_i - T^n_i}{\Delta t}=\alpha \frac{T^{n+1}_{i-1}-2T^{n+1}_i+T^{n+1}_{i+1}}{\Delta x^2}
\end{align*}

In matrix notation this is equivalent to:

\begin{equation*}
    (I-A)\boldsymbol{T}^{n+1} = \boldsymbol{T}^{n}\; \; \Leftrightarrow \; \; (I-A)^{n+1}\boldsymbol{T}^{n+1} = \boldsymbol{T}^{0},
\end{equation*}

where $I$ is the identity matrix. If we adopt Dirichlet boundary conditions, the matrix $A$ is identical to its counterpart for the forward Euler scheme (see previous notebook). For reference, the eigenvalues $m_k$ of $A$ are real and negative:

\begin{equation*}
m_k = 2\frac{\alpha dt}{dx^2}\left(\cos\left(\frac{\pi k}{nx-1}\right)-1\right),\; k=1,\ldots, nx-2.
\end{equation*}

If we diagonalize $A$ and denote by $\boldsymbol z=(z_1,\ldots,z_{nx-2})$ the coordinates of $\boldsymbol T$ in the basis of eigenvectors, we have:

\begin{equation*}
(I-\Lambda)^{n+1} \boldsymbol z^{n+1} = \boldsymbol z^0 
\end{equation*}

where $\Lambda$ is the diagonal matrix containing the eigenvalues of $A$. Each of the $nx-2$ decoupled equations may therefore be written as:

\begin{equation*}
z_k^{n+1} = \left(\frac{1}{1-m_k}\right)^{n+1} z_k^0 
\end{equation*}

Since $m_k < 0\; \forall k$, all the coordinates $z_k$ remain bounded and the algorithm is unconditionally stable.

We can also examine the stability of the algorithm for the case of periodic boundary conditions using the modified wavenumber analysis. We adopt the same notations as in the previous notebook and get the same expression for the evolution of the Fourier modes,

\begin{align}
\label{eq:matHeatBackwardEulerFourier}
    \frac{d\hat T(k_m,t)}{dt}&=\alpha\left(\frac{e^{-ik_m \Delta x} - 2 + e^{ik_m \Delta x}}{\Delta x^2}\right) \hat{T}(k_m,t) \\
    &= -\alpha k_m'^2 \hat T(k_m,t) \nonumber
\end{align}

with the modified wavenumbers being real and defined by,

\begin{equation*}
k_m'^2 = \frac{2}{\Delta x^2}(1 - \cos(k_m \Delta x)).
\end{equation*}

Because all the coefficients $-\alpha k_m'^2$ are real and negative, the Fourier components remain bounded if we discretize eq. \ref{eq:matHeatBackwardEulerFourier} using the backward time integration method. The algorithm is therefore also unconditionally stable in the case of periodic boundary conditions.

+++

## Crank-Nicolson method

A popular method for discretizing the diffusion term in the heat equation is the Crank-Nicolson scheme. It is a second-order accurate implicit method that is defined for a generic equation $y'=f(y,t)$ as:

\begin{align*}
\frac{y^{n+1} - y^n}{\Delta t} = \frac12(f(y^{n+1}, t^{n+1}) + f(y^n, t^n)).
\end{align*}

You should check that this method is indeed second-order accurate in time by expanding $f(y^{n+1}, t^{n+1})$ in Taylor series.

For the heat equation, the Crank-Nicolson method yields the following expression:

\begin{align}
\label{eq:heatCN}
\frac{T^{n+1}_i - T^n_i}{\Delta t}=\frac{\alpha}{2} \left(\frac{T^{n+1}_{i-1}-2T^{n+1}_i+T^{n+1}_{i+1}}{\Delta x^2}  + 
\frac{T^{n}_{i-1}-2T^{n}_i+T^{n}_{i+1}}{\Delta x^2}\right)
\end{align}

In matrix form (assuming Dirichlet boundary conditions), this is equivalent to:

\begin{equation*}
    (I-A_{cn})\boldsymbol{T}^{n+1} = (I+A_{cn})\boldsymbol{T}^{n}.
\end{equation*}

with $A_{cn}$ being the same matrix as in the previous section except for the prefactor that is now $\displaystyle\frac{\alpha \Delta t}{2\Delta x^2}$.

Both sides of the equation can be diagonalized using the same eigenvectors. Therefore, the coordinates of $\boldsymbol{T}$ in the basis of eigenvectors evolve according to:

\begin{equation*}
    (I-\frac{\Lambda}{2}) \boldsymbol z^{n+1} = (I+\frac{\Lambda}{2})\boldsymbol z^n \Rightarrow \boldsymbol z^{n}_k = \left(\frac{1+m_k/2}{1-m_k/2}\right)^n z^0_k.
\end{equation*}

Since $m_k < 0\; \forall k$, all the coordinates $z_k$ remain again bounded and the algorithm is unconditionally stable.

In the case of periodic boundary conditions, we may again study the stability of the method by decomposing the temperature field in Fourier modes. According to eq. \ref{eq:heatCN} we get:

\begin{align*}
    \frac{\hat T^{n+1}(k_m)-\hat T^{n}(k_m)}{\Delta t}= \frac{\alpha}{2\Delta x^2}((&2\cos(k_m\Delta x)-2)\hat T^{n+1}(k_m) \\ &+(2\cos(k_m\Delta x)-2)\hat T^{n}(k_m))
\end{align*}

or equivalently

\begin{equation*}
    \hat T^{n+1}(k_m) = \frac{1-\frac{\alpha \Delta t}{\Delta x^2}(1-\cos(k_m\Delta x))}{1+\frac{\alpha \Delta t}{\Delta x^2}(1-\cos(k_m\Delta x))}\hat T^{n}(k_m).
\end{equation*}

As the denominator is always larger than the numerator, all the Fourier modes remain bounded and we conclude again the algorithm is unconditionally stable.

Let's use these implicit methods and compare them with the forward Euler method that we used in the previous notebook.

+++

## Numerical solution

To test the above numerical methods we use the same example as in the previous notebook. 
The source term in eq. \ref{eq:1Dheat2} is $\sigma = 2\sin(\pi x)$ and the initial condition is $T_0(x) = \sin(2\pi x)$. To make the algorithms work a bit more, we increase the diffusivity parameter to $\alpha=0.1$. The exact solution of the equation is,

\begin{equation}
T(x,t)=e^{-4\pi^2\alpha t}\sin(2\pi x) + \frac{2}{\pi^2\alpha}(1-e^{-\alpha\pi^2 t})\sin(\pi x).
\end{equation}

To get started we import some helper functions. The corresponding modules are part of the course's `module` directory and its path has to be added to the Python search path. The only exception is the `pde_module` that is located in the current notebook's directory, as it is not useful for other notebooks.

```{code-cell} ipython3
import sys
sys.path.insert(0, '../modules')

# Euler step functions
from steppers import euler_step

# Function to compute an error in L2 norm
from norms import l2_diff

# Function to compute d^2 / dx^2 with second-order 
# accurate centered scheme
from matrices import d2_mat_dirichlet

# Function for the RHS of the heat equation and
# exact solution of our example problem
from pde_module import rhs_heat_centered, exact_solution
```

Physical and grid parameters to achieve a precision smaller than $10^{-6}$ in L2-norm:

```{code-cell} ipython3
# Physical parameters
alpha = 0.1                    # Heat transfer coefficient
lx = 1.                        # Size of computational domain
ti = 0.0                       # Initial time
tf = 5.0                       # Final time

# Grid parameters
nx = 513                       # number of grid points
dx = lx / (nx-1)               # grid spacing
x = np.linspace(0., lx, nx)    # coordinates of grid points
```

To evaluate how much faster we can get our solution with the implicit scheme, we will time our algorithms. The first one we consider is the forward Euler scheme using the Euler step function. We know that in that case, we have to use a Fourier number $F<0.5$ (see the notebook `04_03_Diffusion_Explicit`).

**Forward Euler method in component form:**

```{code-cell} ipython3
# Time parameters
fourier = 0.49                      # Fourier number
dt = fourier*dx**2/alpha            # time step
nt = int((tf-ti) / dt)              # number of time steps

# Solution parameters
T0 = np.sin(2*np.pi*x)              # initial condition
source = 2*np.sin(np.pi*x)          # heat source term
sol = exact_solution(x, tf, alpha)  # Exact solution
```

```{code-cell} ipython3
T = np.empty((nt+1, nx)) # Allocate storage for the solution
T[0] = T0.copy()         # Set the initial condition
```

```{code-cell} ipython3
%%timeit
for i in range(nt):
    T[i+1] = euler_step(T[i], rhs_heat_centered, dt, dx, alpha, source)
```

```{code-cell} ipython3
diff_exact = l2_diff(T[-1], sol)
print(f'The L2-error made in the computed solution is {diff_exact}')
print(f'Time integration required {nt} steps')
```

Using $nx=513$, the accuracy is below our set target but it requires $267493$ time steps.

+++

**Forward Euler method in matrix form**

For the implicit methods, we need to perform matrix multiplications to time advance the solution. As an extra test, we also evaluate the efficiency of the forward Euler scheme in matrix form to assess the time penalty required by the matrix multiplications.

In this case, the time stepping is performed with:
\begin{equation}
    \boldsymbol{T}^{n+1} = (I+A)\boldsymbol{T}^{n}
\end{equation}

The needed matrices are computed as follows:

```{code-cell} ipython3
# d^2 / dx^2 matrix with Dirichlet boundary conditions
D2 = d2_mat_dirichlet(nx, dx)     

# I+A matrix
M = np.eye(nx-2) + alpha*dt*D2
```

Timing of the solution:

```{code-cell} ipython3
T[0] = T0.copy()         # Set the initial condition
```

```{code-cell} ipython3
%%timeit
for i in range(nt):
    T[i+1, 1:-1] = np.dot(M, T[i, 1:-1]) + source[1:-1]*dt

# Set the boundary values
T[-1,0] = 0.
T[-1,-1] = 0.
```

```{code-cell} ipython3
diff_exact = l2_diff(T[-1], sol)
print(f'The L2-error made in the computed solution is {diff_exact}')
print(f'Time integration required {nt} steps')
```

As expected, the algorithm in matrix form gives nearly identical results as in component form. The tiny differences are due to round-off errors. 

We note that the algorithm implemented in a matrix form is two times slower than that implemented in a component form.

Let's turn our attention to the implicit algorithms. For these, there is no limit on the time step but we still want to achieve the desired precision.

**Backward Euler method**

For this method, we have determined empirically that the desired precision is achieved by setting the Fourier number to $F=10$. To check this, let's apply the algorithm:

```{code-cell} ipython3
fourier = 10                   # Fourier number
dt = fourier*dx**2/alpha       # time step
nt = int((tf-ti) / dt)         # number of time steps

# I-A matrix
M = np.eye(nx-2) - alpha*dt*D2
Minv = np.linalg.inv(M)
```

```{code-cell} ipython3
T = np.empty((nt+1, nx)) # Allocate storage for the solution    
T[0] = T0.copy()         # Set the initial condition
```

Timing of the solution:

```{code-cell} ipython3
%%timeit
for i in range(nt):
    T[i+1, 1:-1] = np.dot(Minv, T[i, 1:-1] + source[1:-1]*dt)

# Set boundary values
T[-1,0] = 0
T[-1,-1] = 0
```

```{code-cell} ipython3
diff_exact = l2_diff(T[-1], sol)
print(f'The L2-error made in the computed solution is {diff_exact}')
print(f'Time integration required {nt} steps')
```

Using the backward Euler method, the number of time steps has been reduced by a factor of 20 and the execution time by a factor 10 compared to the forward Euler method in component form!

+++

**Crank-Nicolson method**

The last method we consider here is the Crank-Nicolson method. This methods is second-order accurate in time so we can expect even better improvement. After some testing, we have determined that the Fourier number could be raised to a value of $55$. Let's check this:

```{code-cell} ipython3
fourier = 55              # Fourier number
dt = fourier*dx**2/alpha  # time step

nt = int((tf-ti)/dt)      # number of time steps

# I-0.5*A matrix + inverse
A = np.eye(nx-2) - 0.5*alpha*dt*D2
Ainv = np.linalg.inv(A)

# I+0.5*A matrix
B = np.eye(nx-2) + 0.5*alpha*dt*D2

# (I+0.5A)^{-1} * (I-0.5*A)
C = np.dot(Ainv, B)
```

```{code-cell} ipython3
T = np.empty((nt+1, nx)) # Allocate storage for the solution    
T[0] = T0.copy()         # Set the initial condition
```

```{code-cell} ipython3
%%timeit
scn = np.dot(Ainv, source[1:-1]*dt)
for i in range(nt):
    T[i+1, 1:-1] = np.dot(C, T[i, 1:-1]) + scn

# Set boundary values
T[-1,0] = 0
T[-1,-1] = 0
```

```{code-cell} ipython3
diff_exact = l2_diff(T[-1], sol)
print(f'The L2-error made in the computed solution is {diff_exact}')
print(f'Time integration required {nt} steps')
```

The number of time steps has been further reduced by a factor of 5 and the execution time is approximately 30 times faster than that with the forward Euler method!

+++

## Summary

+++

In this notebook we have discussed implicit discretization techniques for the the one-dimensional heat equation. These are particularly useful as explicit scheme requires a time step scaling with $dx^2$. We have shown that the backward Euler and Crank-Nicolson methods are unconditionally stable for this problem. Although they require more complex arithmetical operations than explicit schemes, they result in very significant reduction in computational cost. These reductions are problem dependent because achieving enough accuracy depends on several factors. However, the discussion indicates that it is always important to explore the various algorithms when trying to solve a numerical problem and there is vast literature available to do this.

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
