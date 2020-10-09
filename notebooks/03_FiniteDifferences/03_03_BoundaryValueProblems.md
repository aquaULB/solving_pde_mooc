---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>
<h1 style="text-align: center">Finite Differences III</h1>


<h2 class="nocount">Contents</h2>

1. [Introduction](#Introduction)
2. [Heated rod](#Heated-rod)

    2.1 [Homogeneous Dirichlet boundary conditions](#Homogeneous-Dirichlet-boundary-conditions)
    
    2.2 [Non-homogeneous Dirichlet boundary conditions](#Non-homogeneous-Dirichlet-boundary-conditions)
    
    2.3 [Neumann boundary conditions](#Neumann-boundary-conditions)
   
3. [Summary](#Summary)


## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

## Heated rod


Let's a consider a rod made of some heat conducting material. Under some simplifying assumptions, the temperature $T(x,t)$ along the rod can be determined by solving the following heat equation based on Fourier's law,

\begin{equation}
    \partial_t T(x,t) = \alpha \frac{d^2 T}{dx^2}(x,t) + \sigma (x,t)
\end{equation}

where $\alpha$ is the thermal conductivity of the rod and $\sigma (x,t)$ some heat source present along the rod. To close this equation, some boundary conditions at both ends of the rod need to be specified. If these boundary conditions and $\sigma$ do not depend on time, the temperature within the rod ultimately settles to the solution of the steady-state equation:

\begin{equation}
 \frac{d^2 T}{dx^2}(x) = b(x), \; \; \; b(x) = -\sigma(x)/\alpha.
\end{equation}

In the examples below, we solve this equation with some common boundary conditions.

To proceed, the equation is discretized on a numerical grid containing $nx$ grid points and the second-order derivative is computed using the centered second-order accurate finite difference formula we described in the previous notebook. Without loss of generality, we assume that the length of the rod is equal to $1$ so that $x\in [0\ 1]$.

If we denote respectively by $T_i$ and $b_i$ the values of $T$ and $b$ at the grid nodes, our discretized equation reads:

\begin{equation}
    A_{ij} T_j = b_i
\end{equation}

where $A_{ij}$ is the discrete analogue of $\frac{d^2 }{dx^2}$.

### Homogeneous Dirichlet boundary conditions

In this first example, we apply homogeneous Dirichlet boundary conditions at both ends of the domain (i.e. the values are set to $0$).

\begin{equation}
 T(0)=0, \; T(1)=0 \; \; \Leftrightarrow \; \; T_0 =0, \; T_{nx-1} = 0.
\end{equation}

The usual way of implementing these boundary conditions with finite differences schemes is to realize that $T_0$ and $T_{nx-1}$ are in fact not unkowns: their values are fixed and the numerical method does not need to solve for them. Our real unknows are $T_i$ with $i \in [1, 2, \dots , nx-3, nx-2]$. In the previous notebook we have defined $A_{ij}$ for the centered second-order accurate second-order derivative as:

\begin{align}
\frac{1}{\Delta x^2}
\begin{pmatrix}
2 & -5 & 4 & -1 & 0 & \dots & 0 & 0 & 0 & 0\\
1 & -2 & 1 & 0 & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & 1 & -2 & 1 & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & -2 & 1 & \dots & 0 & 0 & 0 & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & 0 & 0 & 0  & \dots & 1 & -2 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & 1 & -2 & 1 & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & 0 & 1 & -2 & 1 \\
0 & 0 & 0 & 0 & \dots & 0 & -1 & 4 & -5 & 2
\end{pmatrix}
\end{align}

Let's see how to modify this matrix to take into account the boundary conditions. First consider the equation centered around grid node $1$. It reads,

\begin{equation}
    \frac{(T_0 - 2T_1+T_2)}{\Delta x^2} = b_1 \label{eq:leftBndDC}
\end{equation}

As $T_0=0$, it can be replaced by,

\begin{equation}
    \frac{- 2T_1+T_2}{\Delta x^2} = b_1
\end{equation}

The next equation - around grid node 2 - reads:

\begin{equation}
    \frac{T_1 - 2T_2 + T_3}{\Delta x^2} = b_2
\end{equation}

For this one, there is nothing to change. The next lines of the system also remain unchanged up to,

\begin{equation}
    \frac{T_{nx-4} - 2T_{nx-3} + T_{nx-2}}{\Delta x^2} = b_{nx-3}
\end{equation}

Taking into accound $T_{nx-1}=0$, the equation around grid node $nx-2$ then becomes:

\begin{equation}
    \frac{T_{nx-3} - 2T_{nx-2}}{\Delta x^2} = b_{nx-2}
\end{equation}

If we collect the above equations back into a matrix system we get:

\begin{align}
\frac{1}{\Delta x^2}
\begin{pmatrix}
-2 & 1 & 0 & \dots & 0 & 0 & 0 & 0\\
1 & -2 & 1 & 0 & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & 1 & -2 & 1 & 0 & \dots & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & -2 & 1 & \dots & 0 & 0 & 0 & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & \dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots & 0 \\
0 & 0 & 0 & 0  & \dots & 1 & -2 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & 1 & -2 & 1 & 0 \\
0 & 0 & 0 & 0 & \dots & 0 & 0 & 1 & -2 & 1 \\
0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 1 & -2
\end{pmatrix}
\begin{pmatrix}
    T_1 \\
    T_2 \\
    \vdots \\
    T_{j-1}\\
    T_j \\
    T_{j+1}\\
    \\
    \vdots \\
    T_{nx-3} \\
    T_{nx-2}
\end{pmatrix}
=
\begin{pmatrix}
    b_1 \\
    b_2 \\
    \vdots \\
    b_{j-1}\\
    b_j \\
    b_{j+1}\\
    \\
    \vdots \\
    b_{nx-3} \\
    b_{nx-2}
\end{pmatrix}
\end{align}

The above system is completely closed in terms of the *real* unknows $T_1,\dots, T_{nx-2}$. The matrix $\tilde A_{ij}$ on the left-hand side has dimensions $(nx-2)\times(nx-2)$. Implementing the boundary conditions has in practice removed one line and on column from the original matrix. This is to be expected as we now have $nx-2$ unknows. The system can be solved by inverting $\tilde A_{ij}$ to get:

\begin{equation}
T_i = \tilde A^{-1}_{ij} b_j
\end{equation}

Inverting matrices numerically is time consuming for large-size matrices. In a later chapter of this course we will explain how to obtain approximate inverses for large systems. Here, we will limit our attention to moderatly sized matrices and rely on a `scipy` routine called `inv` (available in the `linalg` submodule). The documentation of this function is available [here][1].

So let's now write a Python code to solve the easiest possible case:

\begin{equation}
 \frac{d^2 T}{dx^2}(x) = -1, \; \; \; T(0)=T(1)=0.
\end{equation}

[1]: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html> "documentation for scipy.linalg.inv"

```python
nx = 41 # number of grid points
lx = 1. # length of invertal
dx = lx / (nx-1) # grid spacing
x = np.linspace(0, 1, nx) # coordinates of points on the grid
b = -1.0*np.ones(nx) # right-side vector at the grid points
T = np.empty(nx) # array to store the solution vector
```

As in the previous notebook, we rely on the `diags`routine to build our matrix.

```python
from scipy.sparse import diags
```

To be able to re-use our code later on, we define a routine to create our matrix modified with the proper boundary conditions:

```python
def d2_mat_dirichlet(nx, dx):
    """
    Constructs the centered second-order accurate second-order derivative
    
    Parameters
    ----------
    nx : integer
        number of grid points
    dx : float 
        grid spacing
    
    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order derivative
        with Dirichlet boundary conditions on both side of the interval
    """
    diagonals = [[1.], [-2.], [1.]] # main diagonal elements
    offsets = [-1, 0, 1] # position of the diagonals entries relative to the main diagonal
    
    # Call to the diags routine; note that diags return a representation of the array;
    # to explicitly obtain its ndarray realisation, the call to .toarray() is needed.
    d2mat = diags(diagonals, offsets, shape=(nx-2,nx-2)).toarray()
    
    # Return the final array divided by the grid spacing **2
    return d2mat / dx**2
```

Let's compute our matrix and check that its entries are what we expect:

```python
A = d2_mat_dirichlet(nx, dx)
print(A)
```

We now import the function to compute the inverse of `d2mat`and act with it on the right-hand side vector $b$. This operation is performed through the `nump.dot` routine that allows many sorts of vector and matrix multiplications. You should have a look at its [documentation page][1].

[1]: <https://numpy.org/doc/stable/reference/generated/numpy.dot.html> "documentation for numpy.dot"

```python
from scipy.linalg import inv
```

```python
Am1 = inv(A) # Compute the inverse of A

# Perform the matrix multiplication of the inverse with the rhs.
# We only the values of b at the interior nodes
T[1:-1] = np.dot(Am1, b[1:-1])

# Manually set the boundary values in the temperature array
T[0], T[-1] = [0, 0]
```

That's it ! If everything went fine, $T$ now contains our solution. We can compare it with the exact solution $T(x)=\frac{1}{2}x(1-x)$ which obviously satisfies the required boundary conditions.

```python
T_exact = 0.5 * x * (1-x) # notice how we multiply numpy arrays pointwise.
```

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x, T_exact, label='Exact solution')
ax.plot(x, T, '^g', label='Computed solution')

ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation - Homogeneous Dirichlet boundary conditions')
ax.legend()
```

### Non-homogeneous Dirichlet boundary conditions


In the above example, we imposed homogeneous Dirichlet boundary conditions at both ends of the domain. What happens if we specify a non-zero value for $T$ at the left and/or right boundary node(s)? We will illustrate this for $T(0)=1$ but all other values or combinations Dirichlet boundary conditions are treated in the same way. If we look back at Eq. \ref{eq:leftBndDC}, we have in full generality:

\begin{equation}
    \frac{(T_0 - 2T_1+T_2)}{\Delta x^2} = b_1
\end{equation}

If we set $T_0=1$, this equation becomes,

\begin{equation}
    \frac{(- 2T_1+T_2)}{\Delta x^2} = b_1 - \frac{1}{\Delta x^2}
\end{equation}

We observe that compared to our previous setup, the left-hand side has not changed. However, the value on the right-hand side (the source term) is modified. So the effect of applying a non-homogeneous Dirichlet boundary condition amounts to changing the right-hand side of our equation.

To solve the problem we can re-use everything we computed so far except that we need to modify $b_1$:

```python
b[1] = b[1] - 1. / dx**2
T[1:-1] = np.dot(Am1, b[1:-1]) # Perform the matrix multiplication of the inverse with the rhs.
T[0], T[-1] = [1., 0.] # We set the boundary values
```

Let's check the numerical solution against the exact solution corresponding the modified boundary conditions: $T(x)=\frac12(x+2)(1-x)$.

```python
T_exact = 0.5 * (x+2) * (1-x) # notice how we multiply numpy arrays pointwise.
```

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x, T_exact, label='Exact solution')
ax.plot(x, T, '^g', label='Computed solution')

ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation - Mixed Dirichlet boundary conditions')
ax.legend()
```

The solution looks just as expected !


### Neumann boundary conditions


The last type of boundary conditions we consider is the so-called Neumann boundary condition for which the derivative of the unknown function is specified at one or both ends. Physically this corresponds to specifying the heat flux entering or exiting the rod at the boundaries. Here we are going to set this flux at the left boundary node and assign a specific temperature at the right boundary node:

\begin{equation}
 T'(0)=2, \; T(1) = 1.
\end{equation}

The condition at the right boundary node is treated in the same way as in the previous section. For the left boundary node we need something different.

We have to introduce a discrete version of the condition $T'(0)=2$. As we are using a second-order accurate finite difference for the operator $\frac{d^2 }{dx^2}$, we also want a second-order accurate finite difference for $\frac{d }{dx}$. Indeed, in many problems, the loss of accuracy used for the boundary conditions would degrade the accuracy of the solution throughout the domain.

At the left boundary node we therefore use the (usual) forward second-order accurate finite difference for $T'$ to write:

\begin{equation}
    T'_0 = \frac{-\frac32 T_0 + 2T_1 - \frac12 T_2}{\Delta x}=2
\end{equation}

If we isolate $T_0$ in the preivous expression we have:

\begin{equation}
    T_0 = \frac43 T_1 - \frac13 T_2 - \frac43 \Delta x.
\end{equation}

This shows that the Neumann boundary condition can be implemented by eliminating $T_0$ from the unknown variables using the above relation. The heat equation around the grid node $1$ is then modified as:

\begin{equation}
    \frac{(T_0 - 2T_1+T_2)}{\Delta x^2} = b_1 \;\; \rightarrow \;\;
    \frac{-\frac23 T_1 + \frac 23 T_2}{\Delta x^2} = b_1 + \frac43 \Delta x.
\end{equation}

The effect of the Neumann boundary condition is two-fold: it modifies the left-hand side matrix coefficients and the right-hand side source term. Around the other grid nodes, there are no further modifications (except around grid node $nx-2$ where we impose the non-homogeneous condition $T(0)=1$.

All the necessary bits of code are now scattered at different places in the notebook. We rewrite some of them to make the algorithm easier to follow:

```python
nx = 41 # number of grid points
lx = 1. # length of invertal
dx = lx / (nx-1) # grid spacing
x = np.linspace(0, 1, nx) # coordinates of points on the grid
b = -1.0*np.ones(nx) # right-side vector at the grid points
T = np.empty(nx) # array to store the solution vector

# We use d2_mat_dirichlet() to create the skeleton of our matrix
A = d2_mat_dirichlet(nx, dx)

# The first line of A needs to be modified for Neumann boundary condition
A[0,0:2] = np.array([-2./3., 2./3.]) / dx**2

# Computation of the inverse matrix
Am1 = inv(A)

# The source term as grid nodes 1 and nx-2 needs to be modified
b[1] = b[1] + 4./(3.*dx)
b[-2] = b[-2] - 1. / dx**2

# Computation of the solution using numpy.dot
T[1:-1] = np.dot(Am1, b[1:-1])

# We set the boundary value at the left boundary node
# based on Neumann boundary condition
T[0] = 4./3.*T[1] - 1./3.*T[2] -4./3. * dx

# We set the boundary value at the right boundary node
# based on non-homogeneous Dirichlet boundary condition
T[-1] = 1
```

Let's compare the numerical solution with the exact solution $T_{exact}=-\frac12(x^2-4x+1)$

```python
T_exact = -0.5 * (x**2 - 4*x + 1.)  # notice how we multiply numpy arrays pointwise.
```

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x, T_exact, label='Exact solution')
ax.plot(x, T, '^g', label='Computed solution')

ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat equation - Mixed boundary conditions')
ax.legend()
```

Once again, the computed solution behaves appropriately !


## Summary

```python
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```python

```
