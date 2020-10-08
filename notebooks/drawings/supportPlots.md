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

```python

```
