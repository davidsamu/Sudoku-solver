#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:52:28 2017

Collection of generic (project-agnostic) functions to set up and analyze
Brian2 simulations.

@author: David Samu
"""

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


# %% Functions to analyze simulation results.

def get_spike_df(spikemon, idx2grp=None):
    """Convert Brian's spike monitor structure to Pandas DataFrame objects."""

    # Create spikes dataframe.
    Spks = pd.DataFrame({'unit': spikemon.i, 'time': spikemon.t},
                        index=range(spikemon.num_spikes))

    if idx2grp is not None:
        Spks['group'] = Spks['unit'].map(idx2grp)

    return Spks


def get_spike_stats(Spks, idx2grp, sim_length):
    """Return spike statistics according to unit grouping."""

    # Create empty spike stat DF.
    ser_idx2grp = pd.Series(idx2grp)
    SpkStats = pd.DataFrame({'group': ser_idx2grp, 'num_spikes': 0},
                            index=ser_idx2grp.index)
    SpkStats.index.name = 'unit'

    # Add number of spikes for each unit that has spiked.
    nspikes = Spks.groupby('unit').size()
    SpkStats.loc[nspikes.index, 'num_spikes'] = nspikes

    # Add firing rate.
    SpkStats['rate'] = SpkStats.num_spikes / sim_length

    return SpkStats


def get_prd_spike_count(spikemon, t1=None, t2=None):
    """Return spike count for each unit in period [t1, t2]."""

    if t1 is None:
        t1 = min(spikemon.t)
    if t2 is None:
        t2 = max(spikemon.t)

    prd_spk_cnt = {idx: ((spks >= t1) & (spks <= t2)).sum()
                   for idx, spks in spikemon.spike_trains().items()}

    prd_spk_cnt = pd.Series(prd_spk_cnt, name='num_spikes')
    prd_spk_cnt.index.name = 'unit'

    return prd_spk_cnt


# %% Functions to plot simulation results.

def plot_raster(tvec, ivec, cols=None, n=None, trng=None, ax=None):
    """Plot spikes on raster plot."""

    # Spike marker size for raster plots.
    wsp, hsp = 2, .8  # vertical bar

    # Init axes.
    if ax is None:
        ax = plt.axes()

    # No spikes to plot -> return.
    if not len(tvec):
        return ax

    # Init number of units and time range.
    if n is None:
        n = max(ivec)
    if trng is None:
        trng = [min(tvec), max(tvec)]

    # Plot raster.
    # Spike markers are plotted in relative size (axis coordinates)
    patches = [Rectangle((t-wsp/2, i+1-hsp/2), wsp, hsp)
               for t, i in zip(tvec, ivec)
               if t >= trng[0] and t <= trng[1]]
    collection = PatchCollection(patches, facecolor=cols, edgecolor=cols)
    ax.add_collection(collection)

    # Format plot.
    ax.set_xlim(trng)
    ax.set_ylim([0.5, n+0.5])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')

    # Hide axis ticks and labels.
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    # Hide requested spines of axes. (And don't change the others!)
    for side in ['bottom', 'left', 'top', 'right']:
        ax.spines[side].set_visible(False)

    # Order units from top to bottom, only after setting axis limits!
    ax.invert_yaxis()

    return ax
