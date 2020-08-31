---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Time integration - Part 2


For convenience, we start by importing some modules needed below:

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('./mainstyle.use')
```

In the previous notebook we have considered the forward Euler scheme to time march ordinary differential equations. We have discussed its accuracy and stability with respect to the size of the time step.

Here we introduce some more accurate methods with larger domains of stability that are therefore applicable in more complex situations. We only consider systems of first order differential equations as problems containing higher order derivatives may be reduced to such systems. Our system of equation thus reads,

\begin{align}
    y'(t)=f(t,y)
\end{align}

where $y$ and $f$ are vector valued functions.

## A two stage Runge-Kutta scheme

The forward Euler method is defined through:

\begin{align}
    & y^{n+1} \equiv y^n + f(t,y^n) dt & (\textrm{Forward Euler method}) \label{eq:ForwardEuler3},
\end{align}

with all the intermediate times denoted $t^n = t+ndt$ and the corresponding values of $y(t)$ as $y^n = y(t^n)$.

Graphically, we see that $y^{n+1}$ is evaluated using the value $y^n$ and the slope (derivative) of $y$ at time $t^n$:

<img src="figures/Euler.png" align="center" width="500">

Runge-Kutta schemes increase the accuracy of the estimated value $y^{n+1}$ by introducing intermediate instants between $t^n$ and $t^{n+1}$ at which the derivative of $y$ is evaluated and by making use of this information.

The following two stage Runge-Kutta method is the simplest of such schemes. Graphically, this scheme is defined as follows:

<img src="figures/RK2.png" align="center" width="500">

so that,

\begin{align}
    y^* = y^n +\frac{dt}{2}f(t^n, y^n) \\
    y^{n+1} = y^n + dt f(t+\frac{dt}{2},y^*) \label{RK2ynp1}
\end{align}


In notebook 1.1, we discussed the fact that the forward Euler method is second order accurate for one time step and first order accurate for a complete time interval. Here, we definitely hope to improve these orders of accuracy. Let's check that this is indeed the case by evaluating the Taylor expansion of $y^{n+1}$ in $\eqref{RK2ynp1}$.

\begin{align}
 y^{n+1} & = y^n + dt f(t^n+\frac{dt}{2},y^n + \frac{dt}{2}f(t^n, y^n)) \nonumber \\
         & = y^n + dt \left[ f(t^n, y^n) + \frac{dt}{2} \partial_t f(t^n, y^n) + \frac{dt}{2} f(t^n, y^n)\partial_y f(t^n,y^n) + O(dt^2) \right] \nonumber \\
         & = y^n + dt y^{'n} + \frac{dt^2}{2}y^{''n} + O(dt^3), \label{TaylorRK2}
\end{align}

where we have used the property: $y''=\partial_t f + f\partial_y f$. Eq. \ref{TaylorRK2} proves that the two stage Runge-Kutta method is third-order for one time step and as a consequence it is second-order accurate for a complete time interval (we leave it as an exercise to show that this two stage Runge-Kutta scheme does not match further the Taylor expansion of $y^{n+1}$ and is therefore not of higher order accuracy). 


Let us now discuss the stability of this two stage Runge-Kutta method for a general autonomous linear system of equations. As usual, we may diagonalize the system defined through the matrix $f$ and write:

\begin{align}
    z' = \Lambda z,
\end{align}

where $\Lambda$ is the diagonal matrix composed of the eigenvalues $\lambda_k$ of $f$ and $z$ are the coordinates of $y$ in the eigenvectors basis.

Using the two stage Runge-Kutta scheme we then have,

\begin{align}
    z^{n} = (I+ dt \Lambda + \frac{dt^2}{2}\Lambda^2) z^{n-1} \; \Leftrightarrow \; z^{n} = (I+ dt \Lambda + \frac{dt^2}{2}\Lambda^2)^n z^0.
\end{align}

All the components of $z^{n}$ will remain finite for $n\rightarrow \infty$ as long as the following relation is satisified for all the eigenvalues $\lambda_k$:

\begin{align}
    \vert 1+\lambda_k dt + \frac{\lambda_k^2 dt^2}{2} \vert < 1.
\end{align}




Let us apply this Runge-Kutta scheme to the problem of a body in free fall. We recall that this problem may be written in the following matrix form:

\begin{align}
\begin{pmatrix}
    h^{n+1} \\
    v^{n+1}
\end{pmatrix}
=
\begin{pmatrix}
    h^{n} \\
    v^{n}
\end{pmatrix}
+&
\begin{pmatrix}
    0 & 1 \\
    0 & 0
\end{pmatrix}
\begin{pmatrix}
    h^{n} \\
    v^{n}
\end{pmatrix}
dt
+
\begin{pmatrix}
    0 \\
    -g
\end{pmatrix}
dt \\
&\Leftrightarrow \nonumber \\
y^{n+1} &= y^n + Ly^n dt + bdt
\end{align}

with $y=(h\;\; v)$, $b=(0\;\; -g)$ and

\begin{align}
L=
\begin{pmatrix}
    0 & 1 \\
    0 & 0
\end{pmatrix}
\end{align}


We copy/paste our previous implementation of the solution using the forward Euler scheme and make the necessary changes to obtaining it with our Runge-Kutta scheme:

```python
# parameters
g = 9.81 # ms^-2, gravitational constant
h0 = 100. # initial height
v0 = 0. # initial speed

t0 = 0. # initial time
tf = 4.0
dt = 0.1
nt = int((tf-t0) / dt) # number of time steps

# Create a numpy array to contain the intermediate values of y,
# including those at ti and tf
y = np.empty((nt+1, 2))

# Store initial condition in y[:,0]
y[0] = np.array([h0, v0])

# Create vector b
b = np.array([0, -g])

# Create matrix L
L = np.array([[0, 1], [0, 0]])

# Perform the time stepping
for i in range(nt):
    y_star = y[i] + 0.5 * (np.dot(L,y[i])*dt + b*dt)
    y[i+1] = y[i] + np.dot(L,y_star)*dt + b*dt
```

Graphically, the solution looks like:

```python
fig, ax = plt.subplots(1, 2, figsize=(9, 4))

# create an array containing the multiples of dt
t = np.arange(nt+1) * dt

ax[0].plot(t,y[:,1])
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$v$')
ax[0].set_title(r'Speed vs time (m/s)')

ax[1].plot(t,y[:,0])
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$h$')
ax[1].set_title(r'Height vs time (m)')
```

In exercise 2, we ask to check the accuracy of your solution.


## General Runge-Kutta schemes


General Runge-Kutta schemes are defined as follows \cite{Hairer1987}:

\begin{align}
 y^{n+1} &= y^n + dt(b_1 k_1 + \cdots + b_s k_s) \nonumber \\
 k_1 & = f(t^n, y^n) \nonumber \\
 k_2 & = f(t^n + c_2 dt, y^n + dta_{21}k_1) \nonumber \\
 k_3 & = f(t^n + c_3 dt, y^n + dt(a_{31}k_1+a_{32}k_2)) \nonumber \\
 & \cdots \nonumber \\
 k_s & = f(t^n + c_s dt, y^n + dt(a_{s1}k_1+\cdots + a_{s,s-1}k_{s-1})) \nonumber
\end{align}

Some constraints are then put on all the coefficients to achieve a given order of accuracy $O(dt^p)$ for $y^{n+1}$. One then says that the $s$-stage Runge-Kutta method is of order $p$.


The construction of higher order Runge-Kutta schemes is in fact quite complicated and has been the subject of a vast literature (some in depth review of the Runge-Kutta methods may be found in \cite{Butcher2008} or \cite{Hairer1987}). There is in fact no systematic way of obtaining order $p$ methods with a minimum number of stages $s$. Up to $p=4$ one can achieve order $p$ with $s=p$. For $p=5$ and $p=6$ one needs at least $s=p+1$ stages. For $p=7$ and $p=8$ the minimum number of stages are respectively $s=9$ and $s=11$. "*Above this, very little is know*" \cite{Butcher1996}.

Here we therefore focus our attention on a general purpose (globally) fourth order Runge-Kutta scheme that is accurate and stable enough for all the problems we consider in the rest of this course (from now on we call it RK4). It was introduced in 1901 by W. Kutta and reads \cite{Kutta1901}:

\begin{align}
    y^{n+1} & = y^n + \frac16 k_1 + \frac13(k_2 + k_3) + \frac16 k_4 \nonumber \\
    k_1 & = dtf(t^n,y^n) \\
    k_2 & = dtf(t^n+\frac{dt}{2},y^n+\frac12 k_1) \\
    k_3 & = dtf(t^n+\frac{dt}{2},y^n+\frac12 k_2) \\
    k_4 & = dtf(t^n+dt,y^n+k_3)
\end{align}

For an autonomous linear system, it is straightforward to prove that this method is indeed fifth order accurate for one time step. After diagonalisation, we have:

\begin{align}
  z^{n+1} & = z^n + \frac16 dt \Lambda z^n + \frac13 dt \Lambda (z^n + \frac12 dt \Lambda z^n ) + \frac13 dt \Lambda (z^n + \frac12 dt \Lambda (z^n + \frac12 dt \Lambda z^n)) \nonumber \\
  & \;\; \;+\frac16 dt \Lambda (z^n + dt \Lambda (z^n + \frac12 dt \Lambda (z^n + \frac12 dt \Lambda z^n))) \nonumber \\
  & = z^n + dt \Lambda z^n + \frac{dt^2}{2}\Lambda^2 z^n + \frac{dt^3}6 \Lambda^3 z^n + \frac{dt^4}{24} \Lambda^4 z^n
\end{align}

The last expression coincides with the Taylor expansion of $z^{n+1}$ up to fourth-order.

In terms of stability, we also see that the RK4 method is stable for a general autonomous linear system as long as all the eigenvalues of the operator $f$ satisfy,

\begin{align}
    \vert 1+\lambda_k dt + \frac{\lambda_k^2 dt^2}{2} + \frac{\lambda_k^3 dt^3}{6} + \frac{\lambda_k^4 dt^4}{24}\vert < 1.
\end{align}

In the following plot, we compare the regions of stability for the various schemes we have already discussed.

```python
nx = 100
ny = 100
x = np.linspace(-3.0, 1.0, nx)
y = np.linspace(-3.0, 3.0, ny)
X, Y = np.meshgrid(x, y)

Z = X + 1j*Y

# Euler
sigma1 = 1 + Z
NORM1 = np.real(sigma1*sigma1.conj())

#RK2
sigma2 = 1 + Z + Z**2/2.
NORM2 = np.real(sigma2*sigma2.conj())

#RK4
sigma4 = 1 + Z + Z**2/2. + Z**3/6. + Z**4/24.
NORM4 = np.real(sigma4*sigma4.conj())
```

```python
fig, ax = plt.subplots(figsize=(8,8))
CS1 = ax.contour(X, Y, NORM1, levels = [1], colors='k')
CS2 = ax.contour(X, Y, NORM2, levels = [1], colors='k')
CS4 = ax.contour(X, Y, NORM4, levels = [1], colors='k')
ax.set_aspect(1)
ax.set_title('Stability regions')
```

## Exercises


**Exercise 1.** Prove that the two stage Runge-Kutta scheme is not fourth-order accurate for one time step.


**Exercise 2.** For the problem of a body in free fall, compare the solution obtained with the two stage Runge-Kutta scheme to the exact solution. Check that the method is second order for a finite time interval.

**Exercise 3.** Solve the problem of a body in free fall using the RK4 method.



