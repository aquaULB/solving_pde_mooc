---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: 'Python 3.7.7 64-bit (''course'': conda)'
    language: python
    name: python37764bitcoursecondacf09dcbfbf8b41f7b84691ece6539f67
---

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('./style.use')
```

## Basic sample

```python
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,-3,-1]], y[[0,-3,-1]])

ax.plot(x[2:8:2], y[2:8:2], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.text(2, 0.5, r'$i-1$', fontsize=18, horizontalalignment='center')
ax.text(4, 0.5, r'$i$', fontsize=18, horizontalalignment='center')
ax.text(6, 0.5, r'$i+1$', fontsize=18, horizontalalignment='center')

fig.savefig('figures/sample.png', dpi=300)
```

## Taylor's theorem

```python
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_axis_off()
ax.axis('equal')

# edges
x=np.asarray([0, 10])
y = 0.0 * x

ax.plot(x, y, marker='|')
ax.text(0, 0.5, r'$a$', fontsize=18, horizontalalignment='center')
ax.text(10, 0.5, r'$b$', fontsize=18, horizontalalignment='center')

# inside
x=np.asarray([5, 7])
y = 0.0 * x

ax.plot(x, y, linestyle='')
ax.text(5, 0.5, r'$x$', fontsize=18, horizontalalignment='center')
ax.text(7, 0.5, r'$x+\Delta x$', fontsize=18, horizontalalignment='center')

fig.savefig('figures/taylor.png', dpi=300)
```

```python

```
