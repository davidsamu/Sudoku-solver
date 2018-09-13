#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:10:45 2017

Sudoku solver spiking NN.

@author: David Samu
"""

from itertools import product

import brian2.numpy_ as np
import pandas as pd

from sudoku.util import my_brian_tools


# %% Table indexing functions.

def sub_square(n, i, j):
    """Return subsquare index of cell (i, j) of n x n table."""

    # Check if data is valid.
    if n < 4:
        raise ValueError('Matrix size has to be at least 4-by-4!')
    if not np.sqrt(n).is_integer():
        raise ValueError('Matrix length is not a square number!')
    if (i < 0) | (j < 0):
        raise ValueError('Some coordinate is less then 0!')
    if (i >= n) | (j >= n):
        raise ValueError('Some coordinate does not fit into matrix!')

    ns = int(np.sqrt(n))  # subsquare side length
    si = int(i/ns)
    sj = int(j/ns)

    return si, sj


def is_in_same_subsquare(n, i1, j1, i2, j2):
    """
    Do cells (i1, j2) and (i2, j2) fall into the same subsquare
    of table with size n x n?
    """

    s1i, s1j = sub_square(n, i1, j1)
    s2i, s2j = sub_square(n, i2, j2)
    in_same_sq = (s1i == s2i) & (s1j == s2j)

    return in_same_sq


# %% Network model indexing functions.

def lin_idx(i, j, k, n):
    """Convert matrix index to linear index."""

    if (i >= n) | (j >= n) | (k >= n):
        raise ValueError('Some coordinate is out of range!')

    idx = int(i*n*n + j*n + k)

    return idx


def mat_idx(idx, n):
    """Convert linear index into matrix index."""

    if idx > n**3:
        raise ValueError('Matrix index is out of range!')

    i = int(idx / n**2)
    j = int((idx % n**2) / n)
    k = int(idx % n)

    return i, j, k


def all_ij_pairs(n):
    """Return all (i, j) pairs for table size n."""

    nv = np.arange(n)
    ij_pairs = list(product(nv, nv))
    return ij_pairs


def all_ijk_triples(n):
    """Return all (i, j, k) triplets for table size n."""

    nv = np.arange(n)
    ijk_triples = list(product(nv, nv, nv))
    return ijk_triples


def all_cell_idx(n, i, j):
    """Return all indexes for table cell (i, j)."""

    cell_idxs = [lin_idx(i, j, k, n) for k in range(n)]
    return cell_idxs


def all_table_cell_idx(n):
    """Return all indexes for each cell of table."""

    table_cell_idxs = {(i, j): all_cell_idx(n, i, j)
                       for i, j in product(range(n), range(n))}
    return table_cell_idxs


def convert_lin_to_matrix_state(lin_state):
    """Convert binary linear state vector to state matrix."""

    n = int(np.cbrt(len(lin_state)))
    ijk_idxs = [mat_idx(idx, n) for idx in np.argwhere(lin_state)]
    mat_idxs = [list(v) for v in zip(*ijk_idxs)]
    M = np.zeros((n, n, n), dtype=int)
    M[mat_idxs] = 1

    return M


# %% Functions to test if solution is correct or not.

def get_max_rate_sim_solution(spikemon, n, t1, t2):
    """Return solution of simulation using maximum rate WTA method."""

    # For each cell, select unit with highest rate and calculate confidence.
    dsolution = {}
    dconfidence = {}
    idx2counts = my_brian_tools.get_prd_spike_count(spikemon, t1, t2)

    for i, j in all_ij_pairs(n):
        idxs = all_cell_idx(n, i, j)
        r = idx2counts[idxs]

        sol = mat_idx(r.argmax(), n)[-1] + 1
        conf = r.max() / r.sum()

        dsolution[i, j] = sol
        dconfidence[i, j] = conf

    M = pd.Series(dsolution).unstack()      # solution matrix
    cM = pd.Series(dconfidence).unstack()   # confidence matrix

    return M, cM


def test_correct_solution_2D(M):
    """
    Function to test if solution is correct, returning detailed results.
    M is a 2D matrix, i.e. cell-wise competition is assumed to have a winner.
    """

    # Check if matrix is valid size.
    M = np.array(M)
    nrow, ncol = M.shape
    if nrow != ncol:
        raise ValueError('Matrix is not square size!')
    if not np.sqrt(nrow).is_integer():
        raise ValueError('Matrix length is not a square number!')

    unique_vals = np.arange(nrow) + 1

    # Test each row.
    row_correct = np.array([np.array_equal(np.unique(M[i, :]), unique_vals)
                            for i in range(nrow)])

    # Test each column.
    col_correct = np.array([np.array_equal(np.unique(M[:, j]), unique_vals)
                            for j in range(ncol)])

    # Test each sub-rectangle.
    nsrow, nscol = int(np.sqrt(nrow)), int(np.sqrt(ncol))
    sub_correct = [[np.array_equal(np.unique(M[(nsrow*i):(nsrow*(i+1)),
                                               (nscol*j):(nscol*(j+1))]),
                                   unique_vals) for j in range(nscol)]
                   for i in range(nsrow)]
    sub_correct = np.array(sub_correct)

    # Collest test results.
    test_res = {'rows': row_correct,
                'cols': col_correct,
                'subs': sub_correct}

    return test_res


def test_correct_solution_3D(M):
    """
    Function to test if solution is correct, returning detailed results.
    M is a 3D matrix, i.e. multiple cells can be active at the same time.

    Only gives approximate results, i.e. number of rules violated, but not
    by how much (e.g. 2 or 5 units are active at the same time)!!!
    """

    nrow, ncol, ncell = M.shape

    # Partial solution matrix (at cells where only one unit is active).
    sM_partial = np.full((nrow, ncol), np.nan)
    for i, j in product(range(nrow), range(ncol)):
        units_on = np.where(M[i, j])[0] + 1
        if len(units_on) == 1:  # cells where exactly one unit is active
            sM_partial[i, j] = units_on[0]

    # Test partial solution matrix.
    test_res = test_correct_solution_2D(sM_partial)

    return test_res


def n_errors(M):
    """Return number of errors in solution matrix."""

    test_res = (test_correct_solution_2D(M) if len(M.shape) == 2 else
                test_correct_solution_3D(M))
    n_error = np.array([(~v).sum() for k, v in test_res.items()]).sum()
    return n_error


def is_correct_solution(M):
    """Function to test if solution is correct."""

    is_correct = (n_errors(M) == 0)
    return is_correct
