# jres_sampling.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 15 May 2022 13:09:29 BST

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(figsize=(5, 3))
ax = fig.add_axes([0, 0, 1, 1])

t1_max = 1
t1_axis_max = 2.1
t1_axlim = 2.15
t2_max = 2
t2_axis_max = 2.1
t2_axlim = 2.15

ax.set_xlim(-t2_axlim, t2_axlim)
ax.set_ylim(-t1_axlim + 2, t1_axlim)
plt.axis("off")

# Axes
# x-axis
ax.plot([-t2_axis_max, t2_axis_max], [0, 0], color='k')
ax.scatter(t2_axis_max, 0, color='k', marker=">")
ax.scatter(-t2_axis_max, 0, color='k', marker="<")
ax.text(t2_axis_max - 0.1, -0.1, "$t_2$", horizontalalignment="center")

# y-axis
ax.plot([0, 0], [0, t1_axis_max], color='k')
ax.scatter(0, t1_axis_max, color='k', marker="^", zorder=10)
ax.text(0.1, t1_axis_max - 0.03, "$t_1$", horizontalalignment="center")


t2_pts = np.linspace(0, t2_max, 50)
t1_pts = np.linspace(0, t2_pts[int(len(t2_pts) / 2) + 2], 10)

# Text labels
ax.text(1, -0.1, "+ve", horizontalalignment="center")
ax.text(-1, -0.1, "-ve", horizontalalignment="center")

t2_pts = np.linspace(-t2_max, t2_max, 99)
t1_pts = np.linspace(0, t2_max, 50)

xx, yy = np.meshgrid(t2_pts, t1_pts)
final_points = ax.scatter(xx, yy, zorder=0, lw=0, s=7)

ax.plot([0, -t2_max], [0, t2_max], color='k')

theta = np.linspace(0, 0.75 * np.pi, 100)
r = 0.2
x = r * np.cos(theta)
y = r * np.sin(theta)
ax.plot(x, y, color='k')
props = {"facecolor": "white", "alpha": 0.7, "linewidth": 0, "boxstyle": "square,pad=0.22"}
ax.text(0.3, 0.5, "$-45^{\\circ}$ signal", bbox=props, transform=ax.transAxes, va="center")

fig.savefig("neg_45.png")
