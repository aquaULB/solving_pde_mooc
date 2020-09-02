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

# Time integration - Part 1

In this part of the course we discuss how to solve ordinary differential equations (ODE). Although their numerical resolution is not the main subject of this course, their study nevertheless allows to introduce very important concepts that are essential in the numerical resolution of partial differential equations (PDE).

The ODEs we consider can be written in the form:

\begin{align}
  y^{(n)}=f(t, y, \dots, y^{(n-1)}) \label{eq:ODE},
\end{align}

where $y=y(t)$ is a function of the variable $t$ and $y^{(n)}$ represents the n-th derivative of $y$ with respect to $t$: 

\begin{align}
  y^{(n)}=\frac{d^n y}{dt^n}.
\end{align}

When $f$ does not depend explicitely on time, we qualify the problem as *autonomous*.

Note that we have used $t$ as the variable on which the unkown function $y$ depends and we will usually refer to it as *time*. However, all the methods we describe in this chapter also apply to other problems in which a given function depends on an independant variable and the corresponding ODE have the form \eqref{eq:ODE}.

As an example and toy problem, let us consider radioactive decay. Imagine we have a sample of material containing $N$ unstable nuclei at a given initial time $t_0$. The time evolution of $N$ then follows an exponential decay law:

\begin{align}
  \frac{dN(t)}{dt}=-\alpha N(t).
\end{align}

where $\alpha>0$ is a constant depending on the type of nuclei present in the material. Of course, we don't really need a computer to solve this equation as its solution is readilly obtained and reads:

\begin{align}
  N(t)=N(t_0)e^{-\alpha t} \label{eq:expDecay}
\end{align}

However, our objective here is to obtain the above time evolution using a numerical scheme.

## The Forward Euler method

The most elementary time integration scheme - we also call these 'time advancement schemes' - is known as the forward Euler method. We will use it to introduce several concepts that will pop up frequently in the rest of course. This scheme is based on computing an approximation of the unknown function at time $t+dt$ from its known value at time $t$ using the Taylor expansion limited to the first two terms. For radioactive decay, we then have:

\begin{align}
   & N(t+dt) \equiv N(t) + N'(t)dt & \textrm{Forward Euler method} \label{eq:ForwardEuler}
\end{align}

From this equation, we note that the forward Euler method is second order for going from $t$ to $t+dt$ (the dropped term in the Taylor expansion is $O(dt^2))$. Once the value of $N$ is known at time $t+dt$, one can re-use \eqref{eq:ForwardEuler} to reach time $t+2dt$ and so on...

Schematically, we therefore start the time marching procedure at the initial time $t_0$ and make a number of steps (called time steps) of size $dt$ until we reach the final desired time $t_f$. In order to do this, we need $n_t = (t_f - t_i)/dt$ steps.

By convention, we denote the different intermediate times as $t^n = t+ndt$ and the corresponding values of $N$ as $N^n = N(t+ndt)$ so that $N^n = N(t^n)$.

The forward Euler scheme is then alternatively written as:

\begin{align}
    & N^{n+1} \equiv N^n + N^{'n} dt & \textrm{Forward Euler method} \label{eq:ForwardEuler2}
\end{align}

Here is a Python implementation of the algorithm:
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

Done! The last entry in the array N now contains an estimate for $N(t_f)$.


### Numerical accuracy of the forward Euler method

Let us compare graphically our numerical values with the exact solution \eqref{eq:expDecay}. For that purpose we again use the matplotlib package:

```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../styles/mainstyle.use')

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
fig.savefig('../figures/radioactiveDecay.png', dpi=300)


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
ax.loglog(dt_list, slope, color='green', label=r'$dt$')

# set plot options
ax.set_xlabel(r'$dt$')
ax.set_ylabel(r'Error')
ax.set_title(r'Accuracy')
ax.legend()
fig.savefig('../figures/eulerSlope.png', dpi=300)
```

Do you notice something 'surprising' in this plot? Earlier we mentioned an accuracy of second order for the forward Euler method but here we observe an accuracy of first order. In fact, there is a straightforward explanation for this. We said "...that the forward Euler method is second order for going from $t$ to $t+dt$". Here we are comparing values after ```nt``` time steps with ```nt = int((tf-t0) / dt```. The total error is proportional to the product of the error made at each time step multiplied by the number of steps. As the latter scales as $dt^{-1}$, the total error scales like $dt^2 / dt = dt$. One says that the error made during one time step *accumulates* during the computation.


### Numerical stability of the forward Euler method

For the radioactive decay equation, the forward Euler method does a decent job: when reducing the time step, the solution converges to the exact solution, albeit only with first order accuracy. Let us now focus on another crucial property of numerical schemes called their stability. 

For our radioactive problem, we first observe that according to equation \eqref{eq:ForwardEuler2} we have:

\begin{align}
    N^{n} &= (1-\alpha dt)N^{n-1}  = (1-\alpha dt)^2 N^{n-2}= \dots = (1-\alpha dt)^{n}N_{0}
\end{align}

This relation implies that $N^n \rightarrow 0$ only if $\vert 1-\alpha dt \vert < 1$. Otherwise, if $\vert 1-\alpha dt \vert > 1$ our numerical solution will *blow up*. In the jargon, one says that the forward Euler scheme is unstable $\vert 1-\alpha dt \vert > 1$. This puts a limit on the time step allowed when performing the numerical integration.

In many problems, the coefficients of the equations considered are complex (e.g. Schrödinger equation). If we generalise our radioactive decay problem to allow for complex valued coefficients $\alpha=\alpha_r + i\alpha_i$, the criteria for stability of the forward Euler scheme becomes,

\begin{align}
  \vert 1-\alpha dt \vert < 1 \Leftrightarrow\qquad (1-\alpha_rdt)^2+(\alpha_idt)^2 < 1
\end{align}

where $\vert \vert$ is the complex norm.

Given this, one can then draw a stability diagram indicating the region of the complex plane $(\alpha_rdt , \alpha_idt)$ where the forward Euler scheme is stable.

```python
# This graph looks really bad, some better styling is needed. Could it be made similar to the one shown in Moin?

fig, ax = plt.subplots()
draw_circle = plt.Circle((-1, 0), 1)

ax.set_aspect(1)
ax.add_artist(draw_circle)

# set plot options
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
ax.set_position([0, 0, 1, 1])
ax.set_xlabel(r'$\alpha_r dt$')
ax.set_ylabel(r'$\alpha_i dt$')
ax.set_title(r'Stability of forward Euler scheme')

fig.savefig('../figures/eulerStabilityMap.png', dpi=300)
```

If $dt$ is chosen sufficiently small so that both $\alpha_rdt$ and $\alpha_i dt$ lie in the disk shown in the plot, the forward Euler scheme will be stable. We see in particular that the forward Euler scheme cannot be made stable if $\alpha$ is purely imaginary, however small we choose the time step (we will consider a consequence of this below).


### Multi-dimensional example

So far we have only considered a simple one dimensional example in which the unkown function is a scalar. In pratice, many problems are modelled with a series of coupled variables and the corresponding equation is multi-dimensional. Multi-dimensional equations also arise when our starting equations contain higher-order derivatives and need to be converted to a system of first-order equations. For example, let's consider an object in free fall. Its acceleration is constant and equal to minus the gravitational constant $g$. We therefore have:

\begin{align}
    \frac{d^2 h}{d t^2}=-g,
\end{align}

where $h$ is the height of the object with respect to the ground. In order to solve this equation, we introduce the velocity $v=\frac{dh}{dt}$ of the object and consider the system:

\begin{align}
    \frac{dh}{dt}=v,\\
    \frac{dv}{dt}=-g. 
\end{align}

If we apply the forward Euler scheme to this system we get:

\begin{align}
    h^{n+1}=h^n + v^n dt,\\
    v^{n+1}=v^n  - g dt 
\end{align}

Note that on the rhs of the equations, all the quantities are at time $t^n$ and are explicitely known and allow to jump to time $t^{n+1}$. One says that the scheme is *explicit*. We will consider *implicit* scheme further in the course. We can also write the system of equation in matrix form:

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

Let's solve this system numerically and use numpy array functionalities to write our solution in a more compact way. As initial condition, we choose $h_0=100\textrm{m}$ and $v_0=0\textrm{m/s}$.

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
    y[i+1] = y[i] + np.dot(L,y[i])*dt + b*dt
```

BK or Liza: Make a comment here on how numpy arrays make the code so much easier to write compared to manipulating each components.

Let's display our results graphically:

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

In exercise 2 we ask you to compare this numerical solutions to the exact solution.


### Numerical stability of the forward Euler method revisited

Let's consider another two dimensional example and analyse the motion of an object attached to a spring. The equation of motion reads:

\begin{align}
    m\frac{d^2 x}{d t^2}=-kx,
\end{align}

where $x$ is the position of the object with respect to its equilibrium position and $k>0$ is the spring constant. Introducing the velocity $v=dx/dt$, this equation is equivalent to the following system:

\begin{align}
    \frac{dx}{dt}=v,\\
    \frac{dv}{dt}=-\gamma^2 x,
\end{align}
with $\gamma =\sqrt{k/m}$.

For the forward Euler scheme we have:

\begin{align}
\begin{pmatrix}
    x^{n+1} \\
    v^{n+1}
\end{pmatrix}
=
\begin{pmatrix}
    x^{n} \\
    v^{n}
\end{pmatrix}
+&
\begin{pmatrix}
    0 & 1 \\
    \gamma^2 & 0
\end{pmatrix}
\begin{pmatrix}
    x^{n} \\
    v^{n}
\end{pmatrix}
dt
\end{align}

It does not seem very different from the previous problem so let's implement this. 

```python
# parameters
k = 2. # spring constant
m = 1. # object's mass
x0 = 0.75 # initial position
v0 = 0. # initial velocity

gamma = np.sqrt(k/m)

t0 = 0. # initial time
tf = 40.0
dt = 0.1
nt = int((tf-t0) / dt) # number of time steps

# Create a numpy array to contain the intermediate values of y,
# including those at ti and tf
y = np.empty((nt+1, 2))

# Store initial condition in y[:,0]
y[0] = np.array([x0, v0])

# Create matrix L
L = np.array([[0, 1], [-gamma**2, 0]])

# Perform the time stepping
for i in range(nt):
    y[i+1] = y[i] + np.dot(L,y[i])*dt
```

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
ax[1].set_ylabel(r'$x$')
ax[1].set_title(r'Position vs time (m)')
```

What's going on ? We know a fritionless harmonic oscillator like the one we are considering here should oscillate back and forth with a constant amplitude. Something has to be wrong in our implementation. Or maybe not...

Let us first compute the eigenvalues $\lambda_i$ and the eigenvectors $v_i$ of the matrix, 

\begin{align}
L=
\begin{pmatrix}
    0 & 1 \\
    \gamma^2 & 0
\end{pmatrix}
\end{align}

We get:

\begin{align}
\lambda_1 = i\gamma,\;\; \lambda_2=-i\gamma,\;\;
v_1 =
\begin{pmatrix}
    1 \\
    i\gamma
\end{pmatrix}
\;\;
v_2 =
\begin{pmatrix}
    1 \\
    -i\gamma
\end{pmatrix}.
\end{align}

The matrix $L$ can then be decomposed as,

\begin{align}
L=Q\Lambda Q^{-1}\; \hbox{with,} \;\;
\Lambda = 
\begin{pmatrix}
    i\gamma & 0 \\
    0 & -i\gamma
\end{pmatrix} \;\; \hbox{and} \;\;
Q=
\begin{pmatrix}
   1 & 1 \\
    i\gamma & -i\gamma
\end{pmatrix}.
\end{align}


Using the vector notation $y=(x\;\; v)$, we can then reformulate our time advancement scheme as,

\begin{align}
y^{n+1} = y^{n}+ Ly^{n}dt\;\; & \Leftrightarrow \;\; Q^{-1}y^{n+1} = Q^{-1}y^{n+1} + Q^{-1}Ly^{n}dt \\
& \Leftrightarrow \;\; z^{n+1} = z^{n+1} + \Lambda z^{n}dt. \label{eq:eigenCoor}
\end{align}

In $\eqref{eq:eigenCoor}$, $z=(z_1\;\; z_2)$ are the coordinates in the eigenvector basis $y=z_1(t) x + z_2(t) v$. In this basis, the system of equation is decoupled and reads:

\begin{align}
    z_1^{n+1} = z_1^{n} + i\gamma z_1^{n} dt\\
    z_2^{n+1} = z_2^{n} - i\gamma z_2^{n} dt
\end{align}

It is now clear why the forward Euler scheme displays the diverging behaviour observed in the plots. The coefficients present in the advancement scheme are both purely imaginery and we have seen above that their product with $dt$ necessarily lie outside of the domain of stability of the scheme. Therefore, we cannot avoid the divergence of our solution by taking even a very small time step. The forward Euler scheme is therefore not adapted to the simulation of a simple harmonic oscillator !


## Summary

In this notebook we have described the foward Euler scheme and how we can discretise an ordinary differential equation (or a system of ODE) to compute the time evolution of the physical quantities under consideration.

We have discussed the accuracy of the scheme and its stability. The former results directly from the number of terms retained in the Taylor expansion of the variables, while the latter originates from the structure of the time advancement scheme and the eigenvalues of the rhs linear operator appearing the discretized equations.

In the next notebook, we introduce some more efficient time advancement schemes which have both better accuracy and larger domains of stability. They are know as Runge-Kutta schemes and we will use them extensively when analysing partial differential equations later on in the course.


## Exercices


**Exercise 1.** Write a Python code and perform the corresponding visualisation showing that for one time step, the forward Euler method is indeed of second order accuracy.

**Exercise 2.** In the case of the body in free fall, compare the solution obtained with the forward Euler scheme to the exact solution. Check again that the method is first order for a finite time interval.

**Exercise 3.** Check that the forward Euler scheme is unstable for the case of the body in free fall when $dt$ is large enough so that the product $\lambda dt$ lies outside the domain of stability. What is the maximum time step allowed?

```python

```
