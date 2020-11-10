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

# One dimensional heat equation

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Explicit-resolution-of-the-1D-heat-equation" data-toc-modified-id="Explicit-resolution-of-the-1D-heat-equation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Explicit resolution of the 1D heat equation</a></span><ul class="toc-item"><li><span><a href="#Matrix-stability-analysis" data-toc-modified-id="Matrix-stability-analysis-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Matrix stability analysis</a></span></li><li><span><a href="#Modified-wave-number-analysis" data-toc-modified-id="Modified-wave-number-analysis-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Modified wave number analysis</a></span></li><li><span><a href="#Numerical-solution" data-toc-modified-id="Numerical-solution-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Numerical solution</a></span></li></ul></li><li><span><a href="#Python-loops---break-and-continue" data-toc-modified-id="Python-loops---break-and-continue-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Python loops - break and continue</a></span></li><li><span><a href="#Convergence-of-the-numerical-solution" data-toc-modified-id="Convergence-of-the-numerical-solution-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Convergence of the numerical solution</a></span></li><li><span><a href="#Exercises" data-toc-modified-id="Exercises-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exercises</a></span></li></ul></div>

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

In the first notebooks of this chapter, we have described several methods to numerically solve the first order wave equation. We showed that the stability of the algorithms depends on the combination of the time advancement method and the spatial discretization.

Here we treat another case, the one dimensional heat equation:

\begin{equation}
\label{eq:1Dheat}
   \partial_t T(x,t) = \alpha \frac{d^2 T} {dx^2}(x,t) + \sigma (x,t).
\end{equation}

where $T$ is the temperature and $\sigma$ is an optional heat source term.

Besides discussing the stability of the algorithms used, we will also dig deeper into the accuracy of our solutions. Up to now we have discussed accuracy from the theoretical point of view and checked that the numerical solutions computed were in qualitative agreement with exact solutions. But this is not enough, we have to quantitatively validate our solutions and check their quality. Several ways to do this are described below.

+++

## Explicit resolution of the 1D heat equation

+++

### Matrix stability analysis

We begin by considering the forward Euler time advancement scheme in combination with the second-order accurate centered finite difference formula for $d^2T/dx^2$. Without the source term, the algorithm then reads:

\begin{align}
\label{heatEulerForward}
& \frac{T^{n+1}_i - T^n_i}{dt}=\alpha \frac{T^n_{i-1}-2T^n_i+T^{n}_{n+1}}{\Delta x^2} \\
&\Leftrightarrow
T^{n+1}_i = T^n_i+\frac{\alpha dt}{\Delta x^2} (T^n_{i-1}-2T^n_i+T^{n}_{n+1})
\end{align}

In matrix notation this is equivalent to:

\begin{equation}
\label{eq:matHeatEuler}
    \boldsymbol{T}^{n+1} = (I+A)\boldsymbol{T}^{n}\; \; \Leftrightarrow \; \; \boldsymbol{T}^{n+1} = (I+A)^{n+1}\boldsymbol{T}^{0},
\end{equation}

where $I$ is the identity matrix.

Let us first study the case with homogeneous Dirichlet boundary conditions: $T_0^m = T_{nx-1}^m=0, \forall m$. This means that our unknowns are $T^m_1,\ldots,u^T_{nx-2}$ and that the matrix $A$ has dimensions $(nx-2)\times (nx-2)$. Its expression is:

\begin{align}
A = \frac{\alpha dt}{dx^2}
\begin{pmatrix}
a & b & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 0\\
c & a & b & 0 & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & c & a & b & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & 0 & 0 & 0  & \dots & c & a & b & 0 & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & c & a & b & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & 0 & c & a & b \\
0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & c & b
\end{pmatrix},
\end{align}

with $a = -2$ and $b = c = 1$. According to the theorem we quoted in the previous notebook concerning Toeplitz matrices, the eigenvalues $m_k$ of $A$ are real and negative:

\begin{equation}
m_k = 2\frac{\alpha dt}{dx^2}\left(\cos\left(\frac{\pi k}{n}\right)-1\right),\; k=1,\ldots, n-1.
\end{equation}

As a consequence, $I+A$ is diagonalizable with eigenvalues $\lambda_k = 1+m_k$. The algorithm is stable if all these eigenvalues satisfy $\vert \lambda_k \vert < 1$ or in other words, if all the eigenvalues of $A$ are within the stability domain of the Euler method. This imposes the following constraint on $dt$:

\begin{equation}
dt<\frac{dx^2}{2\alpha} \Leftrightarrow  F<0.5
\end{equation}

where $F= \alpha dt/dx^2$ is sometimes referred to as the Fourier number. This condition is quite strict as the limitation on the time step is proportional to $dx^2$.  Explicit integration of the heat equation can therefore become problematic and implicit methods might be rapidly preferred if a high spatial resolution is needed.

If we use the RK4 method instead of the Euler method for the time discretization, eq. \ref{eq:matHeatEuler} becomes,

\begin{equation}
\label{eq:matHeatRK4}
   \boldsymbol{T}^{n+1} = \left(I+A+\frac12 A^2 + \frac16 A^3+A^4\right)^{n+1}\boldsymbol{T}^{0}.
\end{equation}

and the algorithm is stable if all the eigenvalues $m_k$ of $A$ are contained in the stability region of the RK4 time integration scheme. The constraint on the time step is then,

\begin{equation}
dt<2.79\frac{dx^2}{4\alpha} \Leftrightarrow  F<0.7
\end{equation}

### Modified wave number analysis

...

### Numerical solution

Let's implement the algorithm \ref{heatEulerForward} and empirically check the above stability criteria. To make the problem more interesting, we include a source term in the equation by setting: $\sigma = \sin(\pi x)$. As initial condition we choose $T_0(x) = \sin(2\pi x)$. In that case, the exact solution of the equation reads,

\begin{equation}
\label{eq:solution1DHeat}
T(x,t)=e^{-4\pi^2 t}\sin(2\pi x) + (1-e^{-\pi^2 t})\sin(\pi x).
\end{equation}

To get started we import the `euler_step` function from the `steppers` module (see notebook 04_01_Advection for further details):

```{code-cell} ipython3
import sys
sys.path.insert(0, '../modules')
from steppers import euler_step, rk4_step
```

The heat equation is solved using the following parameters. Note that we currently have set $F=0.49$.

```{code-cell} ipython3
# Physical parameters
alpha = 1.                     # Heat transfer coefficient
lx = 1.                        # Size of computational domain

# Grid parameters
nx = 21                        # number of grid points 
dx = lx / (nx-1)               # grid spacing
x = np.linspace(0., lx, nx)    # coordinates of grid points

# Time parameters
t_i = 0.                       # initial time
t_f = 1.                       # final time
fourier = 0.49                 # Fourier number
dt = fourier*dx**2/alpha       # time step
nt = int((t_f-t_i) / dt)       # number of time steps

# Initial condition
T0 = np.sin(2*np.pi*x)     # initial condition
source = 2*np.sin(np.pi*x)       # heat source term
```

In the function below we discretize the right-hand side of the heat equation using the centered finite difference formula of second-order accuracy:

```{code-cell} ipython3
def rhs_centered(T, dx, alpha, source):
    """Returns the right-hand side of the 1D heat
    equation based on centered finite differences
    
    Parameters
    ----------
    T : array of floats
        solution at the current time-step.
    dx : float
        grid spacing
    alpha : float
        heat conductivity
    source : array of floats
        source term for the heat equation
    
    Returns
    -------
    f : array of floats
        right-hand side of the heat equation with
        Dirichlet boundary conditions implemented
    """
    nx = T.shape[0]
    f = np.empty(nx)
    
    f[1:-1] = alpha/dx**2 * (T[:-2] - 2*T[1:-1] + T[2:]) + source[1:-1]
    f[0] = 0.
    f[-1] = 0.
    
    return f
```

We also define a function that return the exact solution at any time $t$:

```{code-cell} ipython3
def exact_solution(x,t):
    """Returns the exact solution of the 1D
    heat equation with heat source term sin(np.pi*x)
    and initial condition sin(2*np.pi*x)
    
    Parameters
    ----------
    x : array of floats
        grid points coordinates
    t: float
        time
    
    Returns
    -------
    f : array of floats
        exact solution
    """
    f = np.exp(-4*np.pi**2*t) * np.sin(2*np.pi*x) + 2.0*(1-np.exp(-np.pi**2*t)) * np.sin(np.pi*x) / np.pi**2
    
    return f
```

And now we perform the time stepping:

```{code-cell} ipython3
T = np.empty((nt+1, nx))
T[0] = T0.copy()

for i in range(nt):
    T[i+1] = euler_step(T[i], rhs_centered, dt, dx, alpha, source)
```

```{code-cell} ipython3
# plot the solution at several times
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, T[0], label='Initial condition')
ax.plot(x, T[int(0.025/dt)], color='green', label='$t=0.05$')
ax.plot(x, T[-1], color='brown', label=f'$t={t_f}$')
ax.plot(x, exact_solution(x, 1.0), '*', label='Exact solution at $t=1$')


ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat transport with forward Euler scheme - forward finite differences')
ax.legend();
```

The solution looks qualitatively very good !

Run the code again with a slightly larger Fourier number equal to $0.51$ and see what happens... The solution blows up and it is clear that the stability criteria needs to be satisfied to get a stable solution.

At the beginning of this notebook we emphasized that we need better ways of checking the accuracy of our numerical solutions. To that end, let us create a function that returns the $L_2$-norm of the difference between two functions:

```{code-cell} ipython3
def l2_diff(f1, f2):
    """
    Computes the l2-norm of the difference
    between a function f1 and a function f2
    
    Parameters
    ----------
    f1 : array of floats
        function 1
    f2 : array of floats
        function 2
    
    Returns
    -------
    diff : float
        The l2-norm of the difference.
    """
    l2_diff = np.sqrt(np.sum((f1 - f2)**2))/f1.shape[0]
    
    return l2_diff
```

Using this function we can compute the relative error - in $L_2$-norm - of our computed solution compared to the exact solution:

```{code-cell} ipython3
error = l2_diff(T[-1], exact_solution(x, t_f)) 
print(f'The L2-error made in the computed solution is {error}')
```

This is already quite good: in terms of the $L_2$ norm, the error is quite small. There are two ways to improve this:

1. Increase the number of grid points
2. Decrease the time step

Before we explore these possibilities, we provide a short tutorial on how to control the flow of python loops and also describe the `while` loop. These two notions are very useful to make our codes more flexible and in particular control the convergence of our numerical solutions.

+++

## Python loops - break and continue

+++

## Convergence of the numerical solution

+++

Now that we know how to control the flow of python loops, let's use this skill to study the convergence of our algorithm in more detail.

Ideally, we would like to specify a maximum error for our solution and change the simulation parameters to meet this threshold. To do so we have to increase the number of grid points and reduce the time step.

In practice we don't have access to the exact solution to measure the error - otherwise why would we bother solving the problem numerically? To overcome this difficulty, we have to perform what is known as a convergence study. The technique relies again on Taylor's theorem. When we design numerical algorithms, we do so in such a way that they have a given convergence rate: as we make the grid spacing and time step smaller and smaller, the solution should get closer and and closer to the exact solution. Therefore, at some point in the refinement process, the obtained solution should barely change as we refine the parameters further. When the difference between two consecutive solutions falls below a given threshold that we choose, we say that we have converged the solution up to a given precision.

Let's rewrite our numerical procedure to implement this strategy. First we define the precision we want to achieve. Strictly speaking, this is not  really a precision but rather a measure of how well the numerical solution is converged:

```{code-cell} ipython3
precision = 1e-6
```

In the current problem, we have to vary two parameters: the grid spacing and the time step. However, we don't have to separately modify the time step as it is computed from the grid spacing to meet the stability criteria. To vary the grid spacing until convergence is met, we will use a `while` loop. This while loop iterates until the L2 difference between two consecutive solutions gets smaller than the precision. However, as we don't know in advance how many grid refinements are needed, we also add a break statement that stops the loop after a given number of grid refinements. This avoids having the computer run for an excessively long time. If that's the case, it might be better to reconsider the discretization scheme and seek a higher order accurate discretization requiring fewer grid points for the same precision.

```{code-cell} ipython3
# Physical parameters
alpha = 1.                     # Heat transfer coefficient
lx = 1.                        # Size of computational domain
t_i = 0.0                      # Initial time
t_f = 1.0                      # Final time
fourier = 0.49                 # Fourier number to ensure stability



max_grid_refinements = 7
grid_refinement = 1

# Starting number of grid points
nx = 3

# Reference solution for comparison after first grid refinement.
# It is arbitrarely set to zero
Tref = np.zeros(nx)

# Initial value for the difference before entering the while loop
# Any value larger than precision is fine
diff = 1.

while (diff > precision):
    
    # Grid parameters
    nx = 2*nx - 1                  # At each grid refinement dx is divided by 2
    dx = lx / (nx-1)               # grid spacing
    x = np.linspace(0., lx, nx)    # coordinates of grid points

    Tn = np.sin(2*np.pi*x)     # initial condition
    source = 2.0*np.sin(np.pi*x)       # heat source term
    
    dt = fourier*dx**2/alpha       # time step
    nt = int((t_f-t_i) / dt)       # number of time steps
    
    # Time stepping
    for i in range(nt):
        Tnp1 = euler_step(Tn, rhs_centered, dt, dx, alpha, source)
        Tn = Tnp1.copy()
    
    # diff computation
    diff = l2_diff(Tnp1[::2], Tref)
    print(f'L2 difference after {grid_refinement} grid refinement(s) (nx={nx}): {diff}')
    
    grid_refinement += 1
    
    if grid_refinement > max_grid_refinements:
        break
    
    # Store the new reference solution
    Tref = Tnp1.copy()
    
if (diff < precision):
    print(f'\nSolution converged with required precision for {nx} grid points')
else:
    print('\nSolution not converged within the maximum allowed grid refinements')
    print(f'Last number of grid points tested: nx = {nx}')
```

The code tells us that by going from $nx=129$ to $nx=257$, the L2-norm of the difference between the solutions has gotten below $10^{-6}$. If that's the precision we want to achieve we now we can stop there and we know we are fine with $nx=257$.

Rerun the above code after setting `precision = 1e-7` and see what happens. You should get a message informing you that the solution could not be converged within the required precision. This means that our break statement was useful; without it, the computations could run for a very long time before stopping.

For reference, we also compare the converged solution (`nx=257`) with the exact solution:

```{code-cell} ipython3
diff_exact = l2_diff(Tnp1, exact_solution(x, t_f))
print(f'The L2-error made in the computed solution is {diff_exact}')
```

The strategy of comparing the solutions while refining the grid worked very well. By doing so, we converged towards the exact solution and the L2 norm of the error is about $10^{-7}$.

+++

## Exercises

**Exercise 1:**

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```{code-cell} ipython3

```
