#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:53:21 2017

@author: David Samu
"""

import brian2.numpy_ as np
from scipy.spatial import distance
from scipy import interpolate

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

from sudoku.util import sudoku_util


# %% Functions to plot Sudoku tables.

def plot_sudoku(M, pM=None, cM=None, add_errors=None, remove_lbls=True,
                title=None, fname=None):
    """
    Plot Sudoku matrix, optionally adding errors.

    M: a complete or partial solution.
    pM: a partial solution, if provided, numbers are colored by differently.
    cM: confidence matrix to scale size of numbers with.
    """

    # Init.
    M = np.array(M)
    if pM is not None:
        pM = np.array(pM)
    if cM is not None:
        cM = np.array(cM)

    nrow, ncol = M.shape
    nsrow, nscol = int(np.sqrt(nrow)), int(np.sqrt(ncol))

    if add_errors is None:
        # Add errors if matrix is complete.
        add_errors = not np.any(np.isnan(M))

    # Init figure.
    base_cell_size = 1
    ndigits_fac = 1 if nrow < 10 else 1.1
    size = ndigits_fac * nrow * base_cell_size
    fig = plt.figure(figsize=(size, size))
    ax = plt.axes()

    # Plot matrix.
    sns.heatmap(M, vmin=0, vmax=0, cmap='OrRd', cbar=False,
                square=True, linecolor='k', linewidth=1, annot=False, ax=ax)

    # Add cell numbers.
    for i, j in sudoku_util.all_ij_pairs(nrow):
        lbl = int(M[i, j]) if not np.isnan(M[i, j]) else ''
        # Color: is cell present in partial solution?
        c = 'k' if pM is None else 'g' if not np.isnan(pM[i, j]) else 'b'
        # Size: confidence level of cell.
        s = 30 if cM is None else 10 + 20 * cM[i, j]
        # Plot cell label.
        ax.text(j+0.5, nrow-i-0.5, lbl, va='center', ha='center',
                weight='bold', fontsize=s, color=c)

    # Remove tick labels.
    if remove_lbls:
        ax.tick_params(labelbottom='off')
        ax.tick_params(labelleft='off')

    # Embolden border lines.
    kws = {'linewidth': 6, 'color': 'k'}
    for i in range(nsrow+1):
        ax.plot([0, ncol], [i*nsrow, i*nsrow], **kws)
    for j in range(nscol+1):
        ax.plot([j*nscol, j*nscol], [0, ncol], **kws)

    # Highlight errors.
    if add_errors:
        col, alpha = 'r', 1./3
        test_res = sudoku_util.test_correct_solution_2D(M)
        # Rows.
        for i in np.where(np.logical_not(test_res['rows']))[0]:
            irow = nrow-i-1
            rect = mpl.patches.Rectangle((0, irow), ncol, 1,
                                         alpha=alpha, fc=col)
            ax.add_patch(rect)
        # Columns.
        for j in np.where(np.logical_not(test_res['cols']))[0]:
            rect = mpl.patches.Rectangle((j, 0), 1, nrow,
                                         alpha=alpha, fc=col)
            ax.add_patch(rect)
        # Sub-squares.
        for i, j in np.argwhere(np.logical_not(test_res['subs'])):
            isrow = nsrow-i-1
            rect = mpl.patches.Rectangle((j*nscol, isrow*nsrow),
                                         nscol, nsrow,
                                         alpha=alpha, fc=col)
            ax.add_patch(rect)

    # Add title.
    if title is not None:
        ax.set_title(title, fontsize='xx-large')

    # Save figure.
    if fname is not None:
        fig.savefig(fname, dpi=300, bbox_inches='tight')

    return ax


# %% Functions to plot connectivity matrices.

def get_node_colors(n):
    """Return node colors for table of size n."""

    cols = sns.color_palette('deep', n)
    return cols


def calc_base_spline(n=50):
    """Calculate base spline (to be used for each connection)."""

    x = np.linspace(0, 1, 4)  # knot points to fit curve onto
    y = [0, 0.05, 0.07, 0]
    xvec = np.linspace(0, 1, n)
    tck = interpolate.splrep(x, y)
    yvec = interpolate.splev(xvec, tck, der=0)

    return xvec, yvec


def calc_3D_spline_coords(v1, v2, xspl, yspl):
    """
    Calculate connecting curve from vector v1 to vector v2 using base pline
    curve.
    """

    # Init.
    n = len(xspl)
    v1, v2 = np.array(v1), np.array(v2)

    # Create connecting straight line.
    x, y, z = [v1[i] + (v2[i] - v1[i]) * np.linspace(0, 1, n)
               for i in (0, 1, 2)]

    # Strictly vertical connection (only z coordinate changes):
    # bend straight link along x axis.
    if (v1[0] == v2[0]) & (v1[1] == v2[1]):
        x = x + (v1[2] - v2[2]) * yspl

    # Any other orientation: bend straight link along z axis.
    else:
        dxy = distance.euclidean(v1[:2], v2[:2])
        sxy = np.sign(v1[0]-v2[0]) if v1[0] != v2[0] else np.sign(v1[1]-v2[1])
        z = z + sxy * dxy * yspl

    return x, y, z


def plot_synapses(S, n, elev_azim_list, fig_dir, nspl=50):
    """
    Visualize Sudoku connectivity S as 3D matrix from different angles.

    TODO: make it weighted!
    """

    # Init params.
    nv = np.arange(n)
    xspl, yspl = calc_base_spline(nspl)

    # Init figure.
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_aspect(1)
    node_cols = get_node_colors(n)

    # Plot table at the bottom.
    kws = {'color': 'k'}
    lims = [-0.5, n-0.5]
    zlvl = [-0.5, -0.5]
    for iv in range(n+1):
        lvl = iv-0.5
        lw = 4 if not iv % np.sqrt(n) else 2
        ax.plot(lims, [lvl, lvl], zlvl, lw=lw, **kws)
        ax.plot([lvl, lvl], lims, zlvl, lw=lw, **kws)

    # Plot nodes.
    x, y, z = zip(*sudoku_util.all_ijk_triples(n))
    all_node_cols = [node_cols[zi] for zi in z]
    ax.scatter(x, y, z, marker='o', c=all_node_cols, s=200)

    # Plot each connection.
    for idx1, idx2 in zip(S.i, S.j):

        i1, j1, k1 = sudoku_util.mat_idx(idx1, n)
        i2, j2, k2 = sudoku_util.mat_idx(idx2, n)

        # Get 3D curve of connection.
        v1, v2 = [i1, j1, k1], [i2, j2, k2]
        x, y, z = calc_3D_spline_coords(v1, v2, xspl, yspl)

        # Plot connection curve.
        ax.plot(x, y, z, ls='-', color=node_cols[k1], alpha=0.5, lw=0.5)

        # TODO: add arrow head to show direction?

    # Format plot.
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    ax.set_zlabel('Neuron')
    for f_tp, f_tl in [(ax.set_xticks, ax.set_xticklabels),
                   (ax.set_yticks, ax.set_yticklabels),
                   (ax.set_zticks, ax.set_zticklabels)]:
        f_tp(nv)
        f_tl(nv+1)

    # Set limits.
    lim = [-0.5, n-0.5]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)

    # Set background color.
    ax.set_facecolor((1.0, 1.0, 1.0))

    # Save it from different viewpoints.
    for elev, azim in elev_azim_list:
        ax.view_init(elev=elev, azim=azim)
        ffig = fig_dir + 'elev_{}_azim_{}.png'.format(elev, azim)
        fig.savefig(ffig, dpi=300, bbox_inches='tight')

    return ax