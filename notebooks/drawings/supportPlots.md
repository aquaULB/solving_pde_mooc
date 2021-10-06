---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Forward-Euler" data-toc-modified-id="Forward-Euler-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Forward Euler</a></span></li><li><span><a href="#RK2" data-toc-modified-id="RK2-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>RK2</a></span></li><li><span><a href="#Stability-map-for-backward-Euler" data-toc-modified-id="Stability-map-for-backward-Euler-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Stability map for backward Euler</a></span></li><li><span><a href="#Stability-map-for-forward-Euler---centered-FD" data-toc-modified-id="Stability-map-for-forward-Euler---centered-FD-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Stability map for forward Euler - centered FD</a></span></li><li><span><a href="#Modified-wave-number" data-toc-modified-id="Modified-wave-number-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modified wave number</a></span></li><li><span><a href="#Logo" data-toc-modified-id="Logo-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Logo</a></span></li><li><span><a href="#2D-grid" data-toc-modified-id="2D-grid-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>2D grid</a></span></li><li><span><a href="#Gauss-Seidel" data-toc-modified-id="Gauss-Seidel-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Gauss-Seidel</a></span></li><li><span><a href="#2D-stencils" data-toc-modified-id="2D-stencils-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>2D stencils</a></span></li></ul></div>
<!-- #endregion -->

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../styles/mainstyle.use')
```

## Forward Euler

```python
fig, ax = plt.subplots(figsize=(8,6))

# Curve
x = np.linspace(0,5,100)
y = ((x+5)/3. - 3.)**3+1.0
dy = ((x+5)/3.-3.)**2
ax.plot(x, y, color='#000000')

# Horizontal locations
a = 30
c = 80

# Vertical lines
ax.axvline(x[a], ymin=0.25, ymax=0.95, ls=':', color='#000000')
ax.axvline(x[c], ymin=0.25, ymax=0.95, ls=':', color='#000000')

# Slopes
slope_a = dy[a]*(x-x[a])+y[a]
ax.plot(x[a-10: c],slope_a[a-10: c])

# Markers
ax.plot(x[c], y[c], marker='o', color='#000000', markersize=5)

ax.plot(x[a], y[a], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[c], slope_a[c], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)

# Text
ax.text(x[a], -1.2, r'$t^n$', horizontalalignment='center')
ax.text(x[c], -1.2, r'$t^{n+1}$', horizontalalignment='center')

ax.text(x[a-7], slope_a[a], r'$y^n$')
ax.text(x[c-11]+0.7, slope_a[c], r'$y^{n+1}$')

# Final formatting
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$y$')
ax.set_title('Forward Euler')

ax.set_xlim(-0.5,5.5)
ax.set_ylim(-2., 3.)

# Save figure. Default format is png.
# dpi defines the resolution in dots per inch.
fig.savefig('../figures/Euler.png', dpi=300)
```

## RK2

```python
fig, ax = plt.subplots(figsize=(8,6))

# Curve
x = np.linspace(0,5,100)
y = ((x+5)/3. - 3.)**3+1.0
dy = ((x+5)/3.-3.)**2
ax.plot(x, y, color='#000000')

# Horizontal locations
a = 30
b = 55
c = 80

# Vertical lines
ax.axvline(x[a], ymin=0.25, ymax=0.95, ls=':', color='#000000')
ax.axvline(x[b], ymin=0.25, ymax=0.95, ls=':', color='#000000')
ax.axvline(x[c], ymin=0.25, ymax=0.95, ls=':', color='#000000')

# Slopes
slope_a = dy[a]*(x-x[a])+y[a]
ax.plot(x[a-10: b], slope_a[a-10: b])

slope_b_star = (dy[b]-.08) * (x-x[b]) + slope_a[b]
ax.plot(x[a: c], slope_b_star[a: c], '--')

slope_b = (dy[b]-.08) * (x-x[a]) + y[a]
ax.plot(x[a: c], slope_b[a: c])

# Markers
ax.plot(x[b], y[b], marker='o', color='#000000', markersize=5)
ax.plot(x[c], y[c], marker='o', color='#000000', markersize=5)

ax.plot(x[a], y[a], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[b], slope_a[b], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[c], slope_b[c], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)

# Text
ax.text(x[a], -1.4, r'$t^n$', horizontalalignment='center')
ax.text(x[b], -1.4, r'$t^n+\frac{dt}{2}$', horizontalalignment='center')
ax.text(x[c], -1.4, r'$t^{n+1}$', horizontalalignment='center')

ax.text(x[a-7], slope_a[a], r'$y^n$')
ax.text(x[b-7], slope_a[b], r'$y^*$', verticalalignment='bottom')
ax.text(x[c-11]+0.7, slope_b[c], r'$y^{n+1}$')

# Arrow
ax.arrow(x[a], slope_b_star[a], 0., slope_b[a] - slope_b_star[a],
         head_width=0.1, head_length=0.1, fc='k', ec='k',
         length_includes_head= True, zorder=4)

# Final formatting
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$y$')
ax.set_title('Second-order Runge-Kutta')

ax.set_xlim(-0.5,5.5)
ax.set_ylim(-2., 2.)

# Save figure. Default format is png.
# dpi defines the resolution in dots per inch.
fig.savefig('../figures/RK2.png', dpi=300)
```

## Stability map for backward Euler

```python
# All these instructions are documented in the EulerMethod notebook
fig, ax = plt.subplots(figsize=(6, 6))

rectangle = plt.Rectangle((-2, -2), 4, 4, fc='green', hatch='/', alpha=0.5)
ax.add_artist(rectangle)

circle = plt.Circle((1, 0), 1, ec='k', fc='white')
ax.add_artist(circle)

ax.set_aspect(1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xmin, xmax = -2.3, 2.3
ymin, ymax = -2., 2.

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)

ax.arrow(xmin, 0., xmax-xmin, 0., fc='k', ec='k', lw=0.5,
         head_width=1./20.*(ymax-ymin), head_length=1./20.*(xmax-xmin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.arrow(0., ymin, 0., ymax-ymin, fc='k', ec='k', lw=0.5,
         head_width=1./20.*(xmax-xmin), head_length=1./20.*(ymax-ymin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.set_xlabel(r'$\lambda_r dt$')
ax.set_ylabel(r'$\lambda_i dt$', rotation=0)

ax.yaxis.set_label_coords(0.6, 0.92)
ax.xaxis.set_label_coords(1.05, 0.475)

ax.set_title('Stability of backward Euler scheme', y=1.01)

ax.set_xticks((-2, 2))
ax.set_yticks((-2, -1, 1))

ax.tick_params(width=2, pad=10)

# fig.savefig('../figures/eulerBackwardStabilityMap.png', dpi=300)

```

## Stability map for forward Euler - centered FD

```python
# Let's configure the size of the figure
# (in inches) to make it a square and bigger.
fig, ax = plt.subplots(figsize=(8, 8))

circle = plt.Circle((-1, 0), 1, ec='k', fc='green', alpha=0.5, hatch='/')

ax.add_artist(circle)

ax.set_aspect(1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xmin, xmax = -2.3, 2.3
ymin, ymax = -2.2, 2.2

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)


ax.arrow(xmin, 0., xmax-xmin, 0., fc='k', ec='k', lw=0.5,
         head_width=1./20.*(ymax-ymin), head_length=1./20.*(xmax-xmin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.arrow(0., ymin, 0., ymax-ymin, fc='k', ec='k', lw=0.5,
         head_width=1./20.*(xmax-xmin), head_length=1./20.*(ymax-ymin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.set_xlabel(r'$\lambda_r\, \frac{cdt}{dx}$', horizontalalignment='right')
ax.set_ylabel(r'$\lambda_i\, \frac{cdt}{dx}$', rotation=0)

ax.yaxis.set_label_coords(0.6, 0.95)
ax.xaxis.set_label_coords(1.05, 0.475)

ax.set_title('Stability domain - Forward Euler - Centered FD', y=1.05)


ax.set_xticks((-2, 1))
ax.set_yticks((-2, -1, 1))


ax.tick_params(width=2, pad=10)


# Now add the eigenvalues for forward Euler - Centered FD
m = 10
k = np.arange(1,m+1)
x = np.zeros(m)
y = -np.cos(np.pi*k/(m+1))

ax.plot(x,y,'o', color='blue')
fig.savefig('../figures/PDEStabilityMap.png', dpi=300)
```

## Modified wave number

```python
# Let's configure the size of the figure
# (in inches) to make it a square and bigger.
fig, ax = plt.subplots(figsize=(8, 8))

circle = plt.Circle((-1, 0), 1, ec='k', fc='green', alpha=0.5, hatch='/')

ax.add_artist(circle)

ax.set_aspect(1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xmin, xmax = -2.3, 2.3
ymin, ymax = -2.2, 2.2

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)


ax.arrow(xmin, 0., xmax-xmin, 0., fc='k', ec='k', lw=0.5,
         head_width=1./20.*(ymax-ymin), head_length=1./20.*(xmax-xmin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.arrow(0., ymin, 0., ymax-ymin, fc='k', ec='k', lw=0.5,
         head_width=1./20.*(xmax-xmin), head_length=1./20.*(ymax-ymin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.set_xlabel(r'$\lambda_r\, dt$', horizontalalignment='right')
ax.set_ylabel(r'$\lambda_i\, dt$', rotation=0)

ax.yaxis.set_label_coords(0.6, 0.95)
ax.xaxis.set_label_coords(1.05, 0.475)

ax.set_title('Stability domain - Coefficients $\lambda_m$ ', y=1.05)


ax.set_xticks((-2, 1))
ax.set_yticks((-1, 1))


ax.tick_params(width=2, pad=10)


# Eigenvalues
L = 1.
nx = 10
dx = L/(nx-1)

m = 10
k = 2*np.pi/L * np.arange(0,m)

# Forward Euler - Centered FD
x = np.zeros(m)
y = -0.8*np.sin(k*dx)

ax.plot(x,y,'o', color='blue', label='Centered')

# Forward Euler - Forward FD
lam = 0.8*(1.-np.exp(1j*k*dx))
ax.plot(lam.real, lam.imag,'o', color='red', label='Forward')

# Forward Euler - Backward FD
lam = 0.8*(np.exp(-1j*k*dx)-1.)
ax.plot(lam.real, lam.imag,'o', color='black', label='Backward')

ax.legend()

fig.savefig('../figures/modified_k.png', dpi=300)
```

## Logo

```python
# Let's configure the size of the figure
# (in inches) to make it a square and bigger.
fig, ax = plt.subplots(figsize=(8, 8))

circle = plt.Circle((-1, 0), 1, ec='k', fc='green', alpha=0.5, hatch='/')

ax.add_artist(circle)

ax.set_aspect(1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xmin, xmax = -2.3, 2.3
ymin, ymax = -2.2, 2.2

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)


ax.arrow(xmin, 0., xmax-xmin, 0., fc='k', ec='k', lw=0.5,
         head_width=1./40.*(ymax-ymin), head_length=1./40.*(xmax-xmin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.arrow(0., ymin+0.2*(ymax-ymin), 0., 0.65*(ymax-ymin), fc='k', ec='k', lw=0.5,
         head_width=1./40.*(xmax-xmin), head_length=1./40.*(ymax-ymin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.yaxis.set_label_coords(0.6, 0.95)
ax.xaxis.set_label_coords(1.05, 0.475)

ax.set_xticks((-2, 1))
ax.set_yticks((-1, 1))
ax.set_axis_off()

ax.tick_params(width=0, pad=10)


# Eigenvalues
L = 1.
nx = 10
dx = L/(nx-1)

m = 10
k = 2*np.pi/L * np.arange(0,m)

# Forward Euler - Centered FD
x = np.zeros(m)
y = -0.8*np.sin(k*dx)

ax.plot(x,y,'o', color='blue')

# Forward Euler - Forward FD
lam = 0.8*(1.-np.exp(1j*k*dx))
ax.plot(lam.real, lam.imag,'o', color='red')

# Forward Euler - Backward FD
lam = 0.8*(np.exp(-1j*k*dx)-1.)
ax.plot(lam.real, lam.imag,'o', color='black')

fig.savefig('../figures/logo.png', dpi=300)
```

## 2D grid

```python
fig, ax = plt.subplots(figsize=(8, 8))

xmin, xmax = -0.1, 10.1
ymin, ymax = -0.1, 10.1
x = y = np.arange(0, 11, 2)
X, Y = np.meshgrid(x, y)

ax.scatter(X, Y)

ax.arrow(xmin-0.5, 0., xmax+2, 0., fc='k', ec='k', lw=0.5,
         head_width=1./40.*(ymax-ymin), head_length=1./40.*(xmax-xmin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.arrow(0., ymin-.5, 0., ymax+2, fc='k', ec='k', lw=0.5,
         head_width=1./40.*(xmax-xmin), head_length=1./40.*(ymax-ymin),
         overhang = 0.3, length_includes_head= True, clip_on = False)

ax.text(xmax+0.6, -0.6, r'$x$', horizontalalignment='center')
ax.text(-0.4, ymax+0.5, r'$y$', horizontalalignment='center')

ax.text(6, 6.5,r'$p_{i,j}$', horizontalalignment='center')
ax.text(6, 8.5,r'$p_{i,j+1}$', horizontalalignment='center')
ax.text(6, 4.5,r'$p_{i,j-1}$', horizontalalignment='center')

ax.text(4, 6.5,r'$p_{i-1,j}$', horizontalalignment='center')
ax.text(4, 8.5,r'$p_{i-1,j+1}$', horizontalalignment='center')
ax.text(4, 4.5,r'$p_{i-1,j-1}$', horizontalalignment='center')

ax.text(8, 6.5,r'$p_{i+1,j}$', horizontalalignment='center')
ax.text(8, 8.5,r'$p_{i+1,j+1}$', horizontalalignment='center')
ax.text(8, 4.5,r'$p_{i+1,j-1}$', horizontalalignment='center')

ax.text(4, -0.6,r'$i-1$', horizontalalignment='center')
ax.text(6, -0.6,r'$i$', horizontalalignment='center')
ax.text(8, -0.6,r'$i+1$', horizontalalignment='center')

ax.text(-0.8, 4,r'$j-1$', horizontalalignment='center')
ax.text(-0.7, 6,r'$j$', horizontalalignment='center')
ax.text(-0.8, 8,r'$j+1$', horizontalalignment='center')

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
ax.set_aspect(1)

fig.savefig('../figures/2Dgrid.png', dpi=300)
```

## Gauss-Seidel

```python
fig, ax = plt.subplots(figsize=(8, 8))

xmin, xmax = -0.1, 10.1
ymin, ymax = -0.1, 10.1
x = y = np.arange(0, 11, 2)
X, Y = np.meshgrid(x, y)

ax.scatter(X, Y)



ax.text(4, -0.6,r'$i-1$', horizontalalignment='center')
ax.text(6, -0.6,r'$i$', horizontalalignment='center')
ax.text(8, -0.6,r'$i+1$', horizontalalignment='center')

ax.text(-0.8, 4,r'$j-1$', horizontalalignment='center')
ax.text(-0.7, 6,r'$j$', horizontalalignment='center')
ax.text(-0.8, 8,r'$j+1$', horizontalalignment='center')

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)
ax.set_axis_off()
ax.set_aspect(1)

#fig.savefig('../figures/GSgrid.png', dpi=300)
```

## 2D stencils

```python
# Five points
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(5, 5))

xmin, xmax = 1, 7
ymin, ymax = 1, 7

ax.set_axis_off()
ax.set_aspect(1)

line = Line2D([2, 6], [4, 4], color='k', axes=ax, zorder=0)
ax.add_line(line)

line = Line2D([4, 4], [2, 6], color='k', axes=ax, zorder=0)
ax.add_line(line)

circle = plt.Circle((4, 4), 0.5, lw=2, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 6), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 2), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((2, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((6, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)

ax.text(4, 4,r'$-4$', ha='center', va='center')
ax.text(4, 6,r'$1$', ha='center', va='center')
ax.text(6, 4,r'$1$', ha='center', va='center')
ax.text(2, 4,r'$1$', ha='center', va='center')
ax.text(4, 2,r'$1$', ha='center', va='center')

ax.text(7, 3,r'$\frac{1}{\Delta x^2}$', ha='center', va='center')

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)

fig.savefig('../figures/2Dstencil5pt.png', dpi=300)
```

```python
# 9 points
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(5, 5))

xmin, xmax = 1, 7
ymin, ymax = 1, 7

ax.set_axis_off()
ax.set_aspect(1)

line = Line2D([2, 6], [4, 4], color='k', axes=ax, zorder=0)
ax.add_line(line)

line = Line2D([4, 4], [2, 6], color='k', axes=ax, zorder=0)
ax.add_line(line)

line = Line2D([2, 6], [2, 6], color='k', axes=ax, zorder=0)
ax.add_line(line)

line = Line2D([2, 6], [6, 2], color='k', axes=ax, zorder=0)
ax.add_line(line)

circle = plt.Circle((4, 4), 0.5, lw=2, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 6), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 2), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((2, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((6, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((2, 2), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((6, 2), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((2, 6), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((6, 6), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)

ax.text(4, 4,r'$-\frac{10}{3}$', ha='center', va='center')
ax.text(4, 6,r'$\frac{2}{3}$', ha='center', va='center')
ax.text(6, 4,r'$\frac{2}{3}$', ha='center', va='center')
ax.text(2, 4,r'$\frac{2}{3}$', ha='center', va='center')
ax.text(4, 2,r'$\frac{2}{3}$', ha='center', va='center')
ax.text(2, 2,r'$\frac{1}{6}$', ha='center', va='center')
ax.text(2, 6,r'$\frac{1}{6}$', ha='center', va='center')
ax.text(6, 2,r'$\frac{1}{6}$', ha='center', va='center')
ax.text(6, 6,r'$\frac{1}{6}$', ha='center', va='center')

ax.text(7, 3,r'$\frac{1}{\Delta x^2}$', ha='center', va='center')

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)

fig.savefig('../figures/2Dstencil9pt.png', dpi=300)
```

```python
# Fourth order
# 9 point
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(8, 8))

xmin, xmax = -1, 9
ymin, ymax = -1, 9

ax.set_axis_off()
ax.set_aspect(1)

line = Line2D([0, 8], [4, 4], color='k', axes=ax, zorder=0)
ax.add_line(line)

line = Line2D([4, 4], [0, 8], color='k', axes=ax, zorder=0)
ax.add_line(line)

circle = plt.Circle((4, 4), 0.5, lw=2, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 6), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 2), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((2, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((6, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((0, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((8, 4), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 0), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)
circle = plt.Circle((4, 8), 0.5, ec='k', fc='white', alpha=1.)
ax.add_artist(circle)

ax.text(4, 4,r'$-5$', ha='center', va='center')
ax.text(4, 6,r'$\frac{4}{3}$', ha='center', va='center')
ax.text(6, 4,r'$\frac{4}{3}$', ha='center', va='center')
ax.text(2, 4,r'$\frac{4}{3}$', ha='center', va='center')
ax.text(4, 2,r'$\frac{4}{3}$', ha='center', va='center')
ax.text(0, 4,r'$\frac{1}{12}$', ha='center', va='center')
ax.text(8, 4,r'$\frac{1}{12}$', ha='center', va='center')
ax.text(4, 0,r'$\frac{1}{12}$', ha='center', va='center')
ax.text(4, 8,r'$\frac{1}{12}$', ha='center', va='center')

ax.text(7, 3,r'$\frac{1}{\Delta x^2}$', ha='center', va='center', fontsize=24)

ax.set_xlim(xmin ,xmax)
ax.set_ylim(ymin, ymax)

fig.savefig('../figures/2Dstencil4th.png', dpi=300)
```

```python

```
