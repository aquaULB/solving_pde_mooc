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

# Time integration

In this part of the course we discuss how to solve what are known as ordinary differential equations (ODE). Although their numerical resolution is not the main subject of this course, they nevertheless allow to introduce very important concepts that are essential in the numerical resolution of partial differential equations (PDE).

The ODEs we consider can be written in the form:

\begin{align}
  y^{(n)}=F(y, \dots, y^{(n-1)}, t) \label{eq:ODE},
\end{align}

where $y=y(t)$ is a function of the variable $t$ and $y^{(n)}$ represents the n-th derivative of $y$ with respect to $t$: 

\begin{align}
  y^{(n)}=\frac{d^n y}{dt^n}.
\end{align}

Note that we have used $t$ as the variable on which the unkown function $y$ depends and we will usually refer to it as *time*. However, all the methods we describe in this chapter also apply to other problems in which a given function depends on an independant variable and the corresponding ODE can be put in the form \eqref{eq:ODE}.

As an example and toy problem, let us consider radioactive decay. Imagine we have a sample of material containing $N$ unstable nuclei at a given initial time $t_0$. The time evolution of $N$ then follows an exponential decay law given by:

\begin{align}
  \frac{dN(t)}{dt}=-\alpha N(t).
\end{align}

where $\alpha>0$ is a constant depending on the type of nuclei present in the material. Of course, we don't really need a computer to solve this equation as its solution is readilly obtained and reads:

\begin{align}
  N(t)=N(t_0)e^{-\alpha t} \label{eq:expDecay}
\end{align}
However, our objective here is to obtain the above time evolution using a numerical scheme.

## Forward Euler method

The most elementary time integration scheme - we also call these 'time advancement schemes' - is known as the forward Euler method. It is based on computing an approximation of the unknown function at time $t+dt$ from its known value at time $t$ using the Taylor expansion limited to the first two terms. For the radioactive decay, we then have:

\begin{align}
   & N(t+dt) \approx N(t) + N'(t)dt + O(dt^2)& \textrm{Forward Euler method} \label{eq:ForwardEuler}
\end{align}

From the above equation, we note that the forward Euler method is second order for going from $t$ to $t+dt$. Once the value of $N$ is known at time $t+dt$, one can re-use \eqref{eq:ForwardEuler} to reach time $t+2dt$ and so on...

Schematically, we therefore start the time marching procedure at the initial time $t_0$ and make a number of steps (called time steps) of size $dt$ until we reach the final desired time $t_f$. In order to do this, we need $n_t = (t_f - t_i)/dt$ steps.
```python
import numpy as np

# parameters
alpha = 0.25 # exponential law coeffecient
t0 = 0.0 # initial time
tf = 5.0 # final time
dt = 0.5 # time step
nt = int((tf-t0) / dt) # number of time steps
N0 = 100

# Create a numpy array to contain the intermediate values of N,
# including those at ti and tf
N = np.empty(nt+1)

# Store initial value in N[0]
N[0] = N0

# Perform the time stepping
for i in range(nt):
    N[i+1] = N[i] - alpha*N[i]*dt
```

Done! The last entry in the array N now contains an estimate for $N(t_f)$. Let us now compare graphically our numerical values with the exact solution \eqref{eq:expDecay}. For that purpose we again use the matplotlib package:

```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('./mainstyle.use')

# create an array containing the multiples of dt
t = np.arange(nt+1) * dt

# Compute exact solution
Nexact = N0*np.exp(-alpha*t)

# plot the exact solution at the different times t
fig, ax = plt.subplots()
ax.plot(t, Nexact, linestyle='-', label=r'Exact solution')
ax.plot(t, N, '^', color='green', label=r'Forward Euler method')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$N$')
ax.set_title(r'Radioactive decay')
ax.legend()
fig.savefig('radioactiveDecay.png', dpi=300)


```

The agreement is not terrible but it does not look excellent either. What is going on? The answer of course comes from the error introduced by cutting the Taylor series in the definition of the forward Euler scheme and we know things should improve if we reduce $dt$ but this will come at the expense of increasing the total number of time steps and the computational cost. To get an idea about this, run the previous code with a smaller and smaller time step and see what happens to the curves.

To analyse this from the quantitative point of view, let us redo the computation using several values of $dt$ and compare the error made in estimating $N(t_f)$. In the following piece of code, we only store the value of $N$ at $t_f$.

```python
# Create a list of different time step (each time step is half of the previous one)
dt = 0.5
dt_list = np.asarray([dt/2**k for k in range(0, 5)])

# create an array to store the values of N(tf) for the different time steps
values = np.empty_like(dt_list)

for i, dt in enumerate(dt_list):
    
    N = N0 # Restore the initial condition
    nt = int((tf-t0) / dt) # number of time steps
    
    for j in range(nt):
        N = N - alpha*N*dt
        
    values[i] = N
```

We now plot the difference between the computed $N(t_f)$ and the exact solution in a log-log plot:

```python
fig, ax = plt.subplots()

# error computation
error = np.abs(values - N0*np.exp(-alpha*tf)) # numpy can substract the same number to the array of values
ax.loglog(dt_list, error, '*', label=r'Error')

# fit a slope to the previous curve
slope = dt_list
ax.loglog(dt_list, slope, color='green', label=r'$dt^{-1}$')

# set plot options
ax.set_xlabel(r'$dt$')
ax.set_ylabel(r'Error')
ax.set_title(r'Accuracy')
ax.legend()
fig.savefig('eulerSlope.png', dpi=300)
```

Do you notice something 'surprising' in this plot? Earlier we mentioned an accuracy of second order for the forward Euler method but here we observe an accuracy of first order. In fact, there is a straightforward explanation for this. We said "...that the forward Euler method is second order for going from $t$ to $t+dt$". Here we are comparing values after ```nt``` time steps with ```nt = int((tf-t0) / dt```. The total error is the product of the error made at each time step multiplied by the number of steps. As the latter scale as $dt^{-1}$, the total error scale like $dt^2 / dt = dt$. One says that error made during one time step *accumulates* during the computation.

As an exercise, write a Python code and perform the corresponding visualisation showing that for one time step, the forward Euler method is indeed of second order accuracy.



**Solution (to hide from students)**

```python
for i, dt in enumerate(dt_list):
    values[i] = N0 - alpha*N0*dt
    
fig, ax = plt.subplots()

# error computation
error = np.abs(values - N0*np.exp(-alpha*dt_list))
ax.loglog(dt_list, error, '*', label=r'Error')

# fit a slope to the previous curve
slope = dt_list**2
ax.loglog(dt_list, slope, color='green', label=r'$dt^{-2}$')

# set plot options
ax.set_xlabel(r'$dt$')
ax.set_ylabel(r'Error')
ax.set_title(r'Accuracy')
ax.legend()
fig.savefig('eulerSlope2.png', dpi=300)
```

# Finite difference discretization

# Ordinary differential equations

# Partial differential equations

# Iterative methods

# Spectral methods

## Fourier transform

## Chebyshev polynomials
```python

```
