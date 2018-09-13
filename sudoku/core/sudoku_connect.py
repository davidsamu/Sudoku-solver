#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:04:36 2017

@author: David Samu
"""

from itertools import product

from sudoku.util import sudoku_util


def create_hardwired_connectivity(S, n):
    """
    Creates hardwired connectivity: in each row, column and subsquare, units
    representing the same number are fully connected with inhibitory synapses,
    each implementing a local soft-winner-take all group.
    """

    # Pre-generate lookup tables to save time in loop.
    ijk_triples = sudoku_util.all_ijk_triples(n)
    ijk2idx = {(i, j, k): sudoku_util.lin_idx(i, j, k, n) 
               for i, j, k in ijk_triples}    
    subsq_lu = {(i1, j1, i2, j2): sudoku_util.is_in_same_subsquare(n, i1, j1, i2, j2)
                for (i1, j1), (i2, j2) in product(sudoku_util.all_ij_pairs(n), repeat=2)}
    
    from_idx = []
    to_idx = []

    for i1, j1, k1 in ijk_triples:        
        for i2, j2, k2 in ijk_triples:

            idx1 = ijk2idx[(i1, j1, k1)]
            idx2 = ijk2idx[(i2, j2, k2)]
            is_same_sq = subsq_lu[(i1, j1, i2, j2)]

            # Within cell:
            # suppress everyone else (but not self).
            if (i1 == i2) & (j1 == j2) & (k1 != k2):
                from_idx.append(idx1)
                to_idx.append(idx2)

            # Along rows or columns, or within sub-square:
            # suppress identical values only.
            elif (k1 == k2) & ((i1 == i2) | (j1 == j2) | is_same_sq):
                from_idx.append(idx1)
                to_idx.append(idx2)

    S.connect(i=from_idx, j=to_idx)

    return S
