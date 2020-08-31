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
    display_name: 'Python 3.7.7 64-bit (''course'': conda)'
    language: python
    name: python37764bitcoursecondacf09dcbfbf8b41f7b84691ece6539f67
---

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('./mainstyle.use')
```

## Forward Euler

```python
fig, ax = plt.subplots(figsize=(8,6))

# Curve
x = np.linspace(0,5,100)
y = -((x+5)/3. - 3.)**2+1.0
dy = -2./3.*((x+5)/3.-3.)
ax.plot(x, y, color='#000000')

# Vertical locations
a = 30
c = 80

# Vertical lines
ax.axvline(x[a], ymin=0.25, ymax=0.95, ls=':', color='#000000')
ax.axvline(x[c], ymin=0.25, ymax=0.95, ls=':', color='#000000')

# Slopes
slope_a = dy[a]*(x-x[a])+y[a]
ax.plot(x[a-10: c],slope_a[a-10: c])

# Markers
ax.plot(x[c], y[c], marker='o', color='#000000')

ax.plot(x[a], y[a], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[c], slope_a[c], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)

# Text
ax.text(x[a], -0.5, r'$t^n$', horizontalalignment='center')
ax.text(x[c], -0.5, r'$t^{n+1}$', horizontalalignment='center')

ax.text(x[a-7], slope_a[a], r'$y^n$')
ax.text(x[c-11], slope_a[c], r'$y^{n+1}$')


# Final formatting
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$y$')
ax.set_title('Forward Euler')

ax.set_xlim(-0.5,5.5)
ax.set_ylim(-1,2)

# Save figure. Default format is png.
# dpi defines the resolution in dots per inch.
fig.savefig('figures/Euler.png', dpi=300)
```

## RK2

```python
fig, ax = plt.subplots(figsize=(8,6))

# Curve
x = np.linspace(0,5,100)
y = -((x+5)/3. - 3.)**2+1.0
dy = -2./3.*((x+5)/3.-3.)
ax.plot(x, y, color='#000000')

# Vertical locations
a = 30
b = 55
c = 80

# Vertical lines
ax.axvline(x[a], ymin=0.25, ymax=0.9, ls=':', color='#000000')
ax.axvline(x[b], ymin=0.25, ymax=0.9, ls=':', color='#000000')
ax.axvline(x[c], ymin=0.25, ymax=0.9, ls=':', color='#000000')

# Slopes
slope_a = dy[a]*(x-x[a])+y[a]
ax.plot(x[a-10: b],slope_a[a-10: b])

slope_b = dy[b] * (x-x[b]) + slope_a[b]
ax.plot(x[b: c],slope_b[b: c])

# Markers
ax.plot(x[b], y[b], marker='o', color='#000000')
ax.plot(x[c], y[c], marker='o', color='#000000')

ax.plot(x[a], y[a], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[b], slope_a[b], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.plot(x[c], slope_b[c], marker='o', markeredgecolor='#cc0000', markeredgewidth=2.5)

# Text
ax.text(x[a], -0.5, r'$t^n$', horizontalalignment='center')
ax.text(x[b], -0.5, r'$t^n+\frac{dt}{2}$', horizontalalignment='center')
ax.text(x[c], -0.5, r'$t^{n+1}$', horizontalalignment='center')

ax.text(x[a-7], slope_a[a], r'$y^n$')
ax.text(x[b-7], slope_a[b], r'$y^*$')
ax.text(x[c-11], slope_b[c], r'$y^{n+1}$')


# Final formatting
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$y$')
ax.set_title('Runge-Kutta 2nd order')

ax.set_xlim(-0.5,5.5)
ax.set_ylim(-1,2)

# Save figure. Default format is png.
# dpi defines the resolution in dots per inch.
fig.savefig('figures/RK2.png', dpi=300)
```

```python

```
