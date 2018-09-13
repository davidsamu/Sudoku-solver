#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:44:08 2017

@author: David Samu
"""

# %% Imports.

import os
import sys

import pickle

import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker

import brian2.numpy_ as np
import brian2.only as br2
from brian2.units import *
from brian2.core.network import TextReport

sys.path.insert(1, '/home/david/Modelling/Sudoku/')

from sudoku.core import sudoku_connect
from sudoku.util import sudoku_util, sudoku_plot, my_brian_tools

proj_dir = '/home/david/Modelling/Sudoku/'

os.chdir(proj_dir)


# %% Some params.

sns.set_style('white')

grp_cols = pd.Series(['b', 'g', 'r'], index=['stim', 'find', 'skip'])


# Do extra plots? Useful for batch processing.
plot_puzzles = False
plot_connectivity = False
plot_test_run = False


# %% Some examples.

n = np.nan

# Size 4x4.
M4p = np.array([[3, 4, 1, 2],
                [n, n, n, n],
                [n, n, n, n],
                [4, 2, 3, 1]])

M4s = np.array([[3, 4, 1, 2],
                [2, 1, 4, 3],
                [1, 3, 2, 4],
                [4, 2, 3, 1]])

# Size 9x9.
M9p = np.array([[5, 3, n, n, 7, n, n, n, n],
                [6, n, n, 1, 9, 5, n, n, n],
                [n, 9, 8, n, n, n, n, 6, n],
                [8, n, n, n, 6, n, n, n, 3],
                [4, n, n, 8, n, 3, n, n, 1],
                [7, n, n, n, 2, n, n, n, 6],
                [n, 6, n, n, n, n, 2, 8, n],
                [n, n, n, 4, 1, 9, n, n, 5],
                [n, n, n, n, 8, n, n, 7, 9]])


M9p = np.array([[n, 3, n, n, n, n, n, n, n],   # Reduced input version.
                [n, n, n, n, n, 5, n, n, n],
                [n, n, 8, n, n, n, n, n, n],
                [n, n, n, n, n, n, n, n, 3],
                [4, n, n, n, n, n, n, n, n],
                [n, n, n, n, 2, n, n, n, n],
                [n, n, n, n, n, n, 2, n, n],
                [n, n, n, 4, n, n, n, n, n],
                [n, n, n, n, n, n, n, 7, n]])


M9s = np.array([[5, 3, 4, 6, 7, 8, 9, 1, 2],
                [6, 7, 2, 1, 9, 5, 3, 4, 8],
                [1, 9, 8, 3, 4, 2, 5, 6, 7],
                [8, 5, 9, 7, 6, 1, 4, 2, 3],
                [4, 2, 6, 8, 5, 3, 7, 9, 1],
                [7, 1, 3, 9, 2, 4, 8, 5, 6],
                [9, 6, 1, 5, 3, 7, 2, 8, 4],
                [2, 8, 7, 4, 1, 9, 6, 3, 5],
                [3, 4, 5, 2, 8, 6, 1, 7, 9]])


# Size 16x16.
M16p = np.array([[1, n, n, 2, 3, 4, n, n, 12, n, 6, n, n, n, 7, n],
                 [n, n, 8, n, n, n, 7, n, n, 3, n, n, 9, 10, 6, 11],
                 [n, 12, n, n, 10, n, n, 1, n, 13, n, 11, n, n, 14, n],
                 [3, n, n, 15, 2, n, n, 14, n, n, n, 9, n, n, 12, n],
                 [13, n, n, n, 8, n, n, 10, n, 12, 2, n, 1, 15, n, n],
                 [n, 11, 7, 6, n, n, n, 16, n, n, n, 15, n, n, 5, 13],
                 [n, n, n, 10, n, 5, 15, n, n, 4, n, 8, n, n, 11, n],
                 [16, n, n, 5, 9, 12, n, n, 1, n, n, n, n, n, 8, n],
                 [n, 2, n, n, n, n, n, 13, n, n, 12, 5, 8, n, n, 3],
                 [n, 13, n, n, 15, n, 3, n, n, 14, 8, n, 16, n, n, n],
                 [5, 8, n, n, 1, n, n, n, 2, n, n, n, 13, 9, 15, n],
                 [n, n, 12, 4, n, 6, 16, n, 13, n, n, 7, n, n, n, 5],
                 [n, 3, n, n, 12, n, n, n, 6, n, n, 4, 11, n, n, 16],
                 [n, 7, n, n, 16, n, 5, n, 14, n, n, 1, n, n, 2, n],
                 [11, 1, 15, 9, n, n, 13, n, n, 2, n, n, n, 14, n, n],
                 [n, 14, n, n, n, 11, n, 2, n, n, 13, 3, 5, n, n, 12]])

M16s = np.array([[1, 5, 10, 2, 3, 4, 9, 11, 12, 16, 6, 14, 15, 13, 7, 8],
                 [14, 16, 8, 13, 5, 15, 7, 12, 4, 3, 1, 2, 9, 10, 6, 11],
                 [9, 12, 4, 7, 10, 16, 6, 1, 8, 13, 15, 11, 3, 5, 14, 2],
                 [3, 6, 11, 15, 2, 13, 8, 14, 7, 5, 10, 9, 4, 16, 12, 1],
                 [13, 4, 14, 3, 8, 7, 11, 10, 5, 12, 2, 6, 1, 15, 16, 9],
                 [8, 11, 7, 6, 4, 1, 2, 16, 9, 10, 14, 15, 12, 3, 5, 13],
                 [12, 9, 1, 10, 13, 5, 15, 6, 3, 4, 16, 8, 14, 2, 11, 7],
                 [16, 15, 2, 5, 9, 12, 14, 3, 1, 11, 7, 13, 10, 6, 8, 4],
                 [6, 2, 16, 14, 11, 9, 4, 13, 15, 1, 12, 5, 8, 7, 10, 3],
                 [7, 13, 9, 1, 15, 2, 3, 5, 11, 14, 8, 10, 16, 12, 4, 6],
                 [5, 8, 3, 11, 1, 10, 12, 7, 2, 6, 4, 16, 13, 9, 15, 14],
                 [15, 10, 12, 4, 14, 6, 16, 8, 13, 9, 3, 7, 2, 11, 1, 5],
                 [2, 3, 5, 8, 12, 14, 10, 15, 6, 7, 9, 4, 11, 1, 13, 16],
                 [10, 7, 13, 12, 16, 3, 5, 9, 14, 8, 11, 1, 6, 4, 2, 15],
                 [11, 1, 15, 9, 6, 8, 13, 4, 16, 2, 5, 12, 7, 14, 3, 10],
                 [4, 14, 6, 16, 7, 11, 1, 2, 10, 15, 13, 3, 5, 8, 9, 12]])


# Collect all puzzles.
Puzzles = {4: M4p, 9: M9p, 16: M16p}
Solutions = {4: M4s, 9: M9s, 16: M16s}


# Plot puzzles.
if plot_puzzles:
    for n in [4, 9, 16]:
        res_dir = 'puzzles/{}x{}/example/'.format(n, n)

        sudoku_plot.plot_sudoku(Puzzles[n],
                                fname=res_dir+'puzzle.png')

        sudoku_plot.plot_sudoku(Solutions[n], Puzzles[n],
                                fname=res_dir+'solution.png')


# %% Table params.

n = 9                 # table side length
N = n*n               # table area
ns = int(np.sqrt(n))  # sub-square side length
nv = np.arange(n)     # array of all side indexes

# Puzzle to solve and (a) solution.
pM = Puzzles[n]
sM = Solutions[n]


n_tot = n*n*n  # total number of neurons in model


# %% Puzzle -> unit mappings.

ij_pairs = sudoku_util.all_ij_pairs(n)
sol_idx = lambda i, j: sudoku_util.lin_idx(i, j, sM[i, j]-1, n)


# Units WITH external input.
input_idx = [sol_idx(i, j) for i, j in ij_pairs
             if not np.isnan(pM[i, j])]
input_vec = np.zeros(n_tot)
input_vec[input_idx] = 1


# Units WITHOUT external input. They can be part of final solution!
no_input_vec = 1 - input_vec
no_input_idx = np.where(no_input_vec)[0]


# Units PART of final solution (including externally stimulated units).
solution_idx = [sol_idx(i, j) for i, j in ij_pairs]
solution_vec = np.zeros(n_tot)
solution_vec[solution_idx] = 1


# Units NOT PART of final solution.
no_solution_vec = 1 - solution_vec
no_solution_idx = np.where(no_solution_vec)[0]


# Units PART of final solution but NOT receiving external input.
solution_no_input_idx = [sol_idx(i, j) for i, j in ij_pairs
                         if np.isnan(pM[i, j])]
solution_no_input_vec = np.zeros(n_tot)
solution_no_input_vec[solution_no_input_idx] = 1


# Unit index --> {stimulated, in solution but not stimulated, not in solution}
# mapping.
idx2grp = {idx: ('stim' if input_vec[idx] else
                 'find' if solution_no_input_vec[idx] else
                 'skip')
           for idx in range(n_tot)}


# %% Define NN model.

# Start a new scope.
br2.start_scope()

# Neuron model equation and parameters.
tau = 10*ms
v0 = 0*mV
v_th = 20*mV

eqs_LIF = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
'''

eqs_IF = '''
dv/dt = 0 : volt (unless refractory)
'''

eqs = eqs_LIF

G = br2.NeuronGroup(n_tot, eqs, threshold='v > v_th', reset='v = v0',
                    refractory=2*ms, method='linear')

# Set unit parameters.
G.v = v0 + (v_th - v0) * np.random.rand(n_tot)  # random initial states


# %% Input parameters.

# Network input params.
bg_freq =  800 * Hz  # background input frequency
in_freq = 2000 * Hz  # stimulated input frequency
Iv = 10 * mV  # strength of excitation by input

# Input: initial configuration of puzzle + some random Poisson input.
freq = bg_freq * np.ones(n_tot)
freq[input_idx] = in_freq

P = br2.PoissonGroup(n_tot, freq)
SP = br2.Synapses(P, G, on_pre='v += Iv')
SP.connect(j='i')


# %% Hardwired synapses.

# *************
# Strength of lateral inhibition of WTA.
# wdict = {4: -60 *mV, 9: -50*mV, 16: -40*mV}  # for stable solutions
wdict = {4: -50 *mV, 9: -40*mV, 16: -30*mV}  # for meta-stable solutions
w = wdict[n]
# *************

S = br2.Synapses(G, G, on_pre='v += w')
S = sudoku_connect.create_hardwired_connectivity(S, n)


# %% Set up monitors and reporting params, store initialized model.

idxs = [input_idx[0], solution_no_input_idx[0], no_solution_idx[0]]
statemon = br2.StateMonitor(G, 'v', record=idxs)
spikemon = br2.SpikeMonitor(G)

report_period = 5*second
file_reporter = TextReport(sys.stderr)

br2.store('init')


# %% Plot synapses [OPTIONAL].

if plot_connectivity:
    conn_dir = proj_dir + 'connectivities/size' + str(n) + '/hardwired/'
    elev_azim_list = [(60, 30), (80, 90), (0, 90), (90, 0)]
    sudoku_plot.plot_synapses(S, n, elev_azim_list, conn_dir)


# %% Run model, analyze results & test solution.

br2.restore('init')  # start with fresh model
S.active = True  # make sure synapses are turned on


# Run simulation.
duration = 10 * second
br2.run(duration, report=file_reporter, report_period=report_period)

# Show profiling results.
# br2.profiling_summary(show=5)

# Create spikes dataframe.
Spks = my_brian_tools.get_spike_df(spikemon, idx2grp)
SpkStats = my_brian_tools.get_spike_stats(Spks, idx2grp, duration)


# Analyze results.
res_dir = 'puzzles/{}x{}/example/'.format(n, n)


# I/F curves.
plt.figure()
for idx, grp in [(0, 'stim'), (1, 'find'), (2, 'skip')]:
    plt.plot(statemon.t/ms, statemon.v[idx]/mV,
             color=grp_cols[grp], label=grp)
plt.xlabel('Time (ms)')
plt.ylabel('v (mV)')
plt.legend()
fname = res_dir + 'example_voltages.png'
plt.savefig(fname, dpi=300, bbox_inches='tight')


# Raster.
plt.figure()
cols = [grp_cols[idx2grp[idx]] for idx in spikemon.i]
my_brian_tools.plot_raster(spikemon.t/ms, spikemon.i, cols)
fname = res_dir + 'raster.png'
plt.savefig(fname, dpi=300, bbox_inches='tight')


# Mean rate of groups over time.
t_start = 0*second
t_end = t_start + duration
tstep = duration / 50
dprdRates = {}
for t1 in np.arange(t_start, t_end, tstep):
    t2 = t1 + tstep
    prdSpks = Spks.loc[(Spks.time >= t1) & (Spks.time < t2)]
    prdSpkStats = my_brian_tools.get_spike_stats(prdSpks, idx2grp, tstep)
    dprdRates[float((t1+t2)/2)] = prdSpkStats[['group', 'rate']]

prdRates = pd.concat(dprdRates).reset_index()
prdRates.rename(columns={'level_0': 'time'}, inplace=True)
plt.figure()
ax = sns.pointplot('time', 'rate', 'group', prdRates,
                   hue_order=grp_cols.index, ci=68, dodge=0.1)
ax.axhline(0, c='k', lw=0.5, ls='--')
plt.legend(loc='best')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#ax.xaxis.set_major_locator(ticker.AutoLocator())
fname = res_dir + 'mean_rates.png'
plt.savefig(fname, dpi=300, bbox_inches='tight')


# Test solution.

# Time interval to use for checking solution.
t1 = max(duration - 1*second, 0*second)
t2 = duration
M, cM = sudoku_util.get_max_rate_sim_solution(spikemon, n, t1, t2)
# sudoku_plot.plot_sudoku(M, pM)
sudoku_plot.plot_sudoku(M, pM, cM)


# Plot proportion of units from each group in final solution.
Mmatches = (M == pd.DataFrame(sM))
init_idxs = ~pd.DataFrame(pM).isnull()
dgrp_sol = {(i, j): {'group': 'init' if init_idxs.loc[i, j] else 'find',
                     'correct': 'correct' if int(Mmatches.loc[i, j]) else 'incorrect'}
            for i, j in sudoku_util.all_ij_pairs(n)}
grp_sol = pd.DataFrame(dgrp_sol).T
grp_sol.index.names = ['i', 'j']

plt.figure(figsize=(6, 3))
ax = plt.axes()
sns.countplot(x='group', hue='correct', data=grp_sol, ax=ax)


# %% Generate table configurations.

duration = 5 * second
t1 = max(duration - 1*second, 0*second)
t2 = duration

n_tables = 100
n_errors = np.full([n_tables], np.nan)
save_tables = False

res_dir = 'puzzles/{}x{}/generated/'.format(n, n)

# Using different random initial conditions.
for irun in range(n_tables):
    print('\n\nRun {}\n'.format(irun+1))

    # Set up network.
    br2.restore('init')  # start with fresh model
    G.v = v0 + (v_th - v0) * np.random.rand(n_tot)  # random initial states
    freq = bg_freq * np.ones(n_tot)  # random uniform Poisson input
    P._rates = P.rates = freq

    # Run network & get solution.
    br2.run(duration, report=file_reporter, report_period=report_period)
    M, cM = sudoku_util.get_max_rate_sim_solution(spikemon, n, t1, t2)

    # Save results.
    if save_tables:
        title = 'Run {}'.format(irun+1)
        fname = '{}tables/{}x{}_run{}.png'.format(res_dir, n, n, irun)
        sudoku_plot.plot_sudoku(M, cM=cM, title=title, fname=fname)

    # Collect stats.
    n_errors[irun] = sudoku_util.n_errors_2D(M)

# Save stats.
fname = '{}stats_n_{}.txt'.format(res_dir, n_tables)
with open(fname, 'w') as fres:

    fres.write('# tables generated: {}, table size: {}x{}\n'.format(n_tables,
                                                                    n, n))

    mean, std = n_errors.mean(), n_errors.std()
    fres.write('\nMean (std) of errors: {} ({})\n'.format(mean, std))

    n_err_str = ', '.join(str(n) for n in n_errors)
    fres.write('\nIndividual runs: {}'.format(n_err_str))


# %% Sampling or rate code?
# Could it be that the network transitions between possible correct solutions,
# but rate-based decoding hides it?

# Need to run this on underdefined problem
# (to allow transition among multiple solutions).


#----------------------------------
# Run simulation.

br2.restore('init')  # start with fresh model
S.active = True  # make sure synapses are turned on

# Run simulation.
duration = 100 * second
br2.run(duration, report=file_reporter, report_period=report_period)

# Show profiling results.
# br2.profiling_summary(show=5)

# Create spikes dataframe.
Spks = my_brian_tools.get_spike_df(spikemon, idx2grp)
SpkStats = my_brian_tools.get_spike_stats(Spks, idx2grp, duration)


#----------------------------------
# Process results.

sigma = 30*ms  # time window after spike when the neuron is considered to be on
select_latest = True  # select latest spiking unit to prevent incompatible states
                      # all neurons are selected when simultaneously firing!

select_groups = sudoku_util.all_table_cell_idx(n)


t1 = 0 * second
t2 = 100 * second

# Init objects.
state_list = []  # list of consequtively visited states
state_idx = {}  # visited state -> state index mapping
Spks_wndw = Spks[(Spks.time >= t1) & (Spks.time <= t2)]  # spike table

# Get visited states and the time the network spent in each.
# Clock-driven implementation.
# Event-driven could be more efficient in some cases?
tstep = br2.defaultclock.dt
tpoints = np.arange(t1+tstep, t2+tstep, tstep)
twindows = pd.DataFrame.from_dict({'tstart': tpoints-sigma, 'tstop': tpoints})
nsteps = len(twindows)
prev_on_units = ()
for i, (ti, tj) in twindows.iterrows():

    # Report progress.
    if (i+1) % 1000 == 0:
        print('{}/{} ({}%)'.format(i+1, nsteps, int(100*(i+1)/nsteps)))

    # Determine current state.
    spks = Spks_wndw[(Spks_wndw.time >= ti) & (Spks_wndw.time <= tj)]
    on_units = tuple(np.sort(spks.unit.unique()))

    # Select only most recently fired unit from each group.
    if select_latest:
        latest_on_units = []
        for cell_ij, cell_units in select_groups.items():  # each group
            cell_on_units = [val for val in on_units if val in cell_units]
            if len(cell_on_units) > 1:  # if more then one unit spiked
                cell_spks = spks[spks.unit.isin(cell_on_units)]
                latest_time = cell_spks.time == cell_spks.time.min()
                latest_on_unit = cell_spks.loc[latest_time].unit
                latest_on_units += list(latest_on_unit)  # add latest one(s)
            else:
                latest_on_units += cell_on_units
        on_units = tuple(latest_on_units)

    # No state change: just update current entry.
    if on_units == prev_on_units:
        state_list[-1]['dur'] += tstep/second

    else:  # State changed: process new state.

        # Add current state to index if not yet added.
        if on_units not in state_idx:
            # Get number of errors in state.
            lin_state = np.zeros(n*n*n)
            lin_state[list(on_units)] = 1
            vM = sudoku_util.convert_lin_to_matrix_state(lin_state)
            nerr = sudoku_util.n_errors(vM)
            # Add new state entry to index.
            state_idx[on_units] = {'idx': len(state_idx), 'n_error': nerr,
                                   'n_active': len(on_units)}

        new_state = state_idx[on_units]
        new_state = {'idx': new_state['idx'],
                     'n_error': new_state['n_error'],
                     'n_active': new_state['n_active'],
                     'tstart': tj, 'dur': tstep/second}
        state_list.append(new_state)

        prev_on_units = on_units

cols = ['idx', 'n_error', 'n_active', 'tstart', 'dur']
state_df = pd.DataFrame(state_list, columns=cols)


#----------------------------------
# Report and plot results.

# Init results folder.
pdir = 'w_{:g}mV_sigma_{:g}ms'.format(w/mV, sigma/ms)
pdir += '_t1_{:g}s_t2_{:g}s_tstep_{:g}us'.format(t1/second, t2/second, tstep/us)
latest_str = 'latest_active' if select_latest else 'fixed_active_window'
res_dir = 'puzzles/{}x{}/state_sampling/{}/{}/'.format(n, n, pdir, latest_str)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Export results.
obj_dict = {'state_df': state_df, 'state_idx': state_idx,
            'Spks': Spks, 'SpkStats': SpkStats}
pickle.dump(obj_dict, open(res_dir+'results.data', 'wb'))


# Plot histogram of firing rates.
plt.figure()
rates = Spks_wndw.groupby('unit').size() / (t2-t1)
ax = sns.distplot(rates, kde=False)
ax.set_yscale('log')
ax.set_xlim([0, None])
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('n')
plt.savefig(res_dir + 'firing_rate_hist.png',
            dpi=300, bbox_inches='tight')

# Plot evolution of number of errors.
plt.figure()
plt.plot(state_df.tstart, state_df.n_error, lw=0.1)
ax = plt.gca()
ax.axhline(0, ls='--', c='grey')
ax.set_xlabel('simulation time (s)')
ax.set_ylabel('number of errors')
plt.savefig(res_dir + 'nerr_timecourse.png',
            dpi=300, bbox_inches='tight')

# Plot evolution of number of active units.
plt.figure()
plt.plot(state_df.tstart, state_df.n_active, lw=0.1)
ax = plt.gca()
ax.axhline(n*n, ls='--', c='grey')
ax.set_xlabel('simulation time (s)')
ax.set_ylabel('number of active units')
plt.savefig(res_dir + 'nact_timecourse.png',
            dpi=300, bbox_inches='tight')

# Plot evolution of transition times.
plt.figure()
plt.plot(state_df.tstart, 1000*state_df.dur, lw=0.1)
ax = plt.gca()
#ax.set_yscale('log')
ax.set_xlabel('simulation time (s)')
ax.set_ylabel('state transition time (ms)')
plt.savefig(res_dir + 'transt_timecourse.png',
            dpi=300, bbox_inches='tight')


# Plot histogram on number of units active.
plt.figure()
bins = np.arange(state_df.n_active.min()-0.5,
                 state_df.n_active.max()+0.5, 1)
ax = sns.distplot(state_df.n_active, bins=bins, kde=False)
ax.axvline(x=n*n, ls='--', c='grey')
ax.set_yscale('log')
ax.set_xlabel('number of active units')
ax.set_ylabel('n')
plt.savefig(res_dir + 'nact_hist.png',
            dpi=300, bbox_inches='tight')


# Scatter between time spent and number of units active.
plt.figure()
ax = sns.regplot(state_df.n_active, 1000*state_df.dur,
                 fit_reg=False, marker='+')
ax.axvline(x=n*n, ls='--', c='grey')
ax.set_xlim([-0.5, None])
ax.set_yscale('log')
ax.set_xlabel('number of active units')
ax.set_ylabel('dur (ms)')
plt.savefig(res_dir + 'nact_dur_scatter.png',
            dpi=300, bbox_inches='tight')


# Plot histogram of time spent in all visited states.
plt.figure()
ax = sns.distplot(1000*state_df.dur, kde=False)
ax.set_yscale('log')
ax.set_xlabel('transition time (ms)')
ax.set_ylabel('n')
ax.set_title('All visited states')
plt.savefig(res_dir + 'transt_hist.png',
            dpi=300, bbox_inches='tight')


# Plot histogram of time spent in visited correct states.
plt.figure()
state_stats_corr = state_df[state_df.n_error == 0]
ax = sns.distplot(1000*state_stats_corr.dur, kde=False)
ax.set_yscale('log')
ax.set_xlabel('transition time (ms)')
ax.set_ylabel('n')
ax.set_title('Visited correct states only')
plt.savefig(res_dir + 'transt_corr_hist.png',
            dpi=300, bbox_inches='tight')


# Plot histogram of errors in visited states.
plt.figure()
bins = np.arange(-0.5, state_df.n_error.max()+1, 1)
ax = sns.distplot(state_df.n_error, bins=bins, kde=False)
ax.set_yscale('log')
ax.set_xlim([0, None])
ax.set_xlabel('number of errors')
ax.set_ylabel('n')
plt.savefig(res_dir + 'nerr_hist.png',
            dpi=300, bbox_inches='tight')


# Scatter between time spent and number of errors across visited states.
plt.figure()
ax = sns.regplot(state_df.n_error, 1000*state_df.dur,
                 x_jitter=0.2, y_jitter=0.0, marker='o')
ax.set_xlim([-0.5, None])
ax.set_xlabel('number of errors')
ax.set_ylabel('dur (ms)')
# Add correlation results.
rval, pval = sp.stats.pearsonr(state_df.n_error, state_df.dur)
txt = 'r = {:.2f} (p = {:.4f})'.format(rval, pval)
ax.text(0.95, 0.95, txt, va='top', ha='right', transform=ax.transAxes)
plt.savefig(res_dir + 'nerr_transt_scatter.png',
            dpi=300, bbox_inches='tight')


# Report some stats.
fname = res_dir + 'solution_stats.txt'
with open(fname, 'w') as fres:

    icorrsts = state_df.index[state_df.n_error == 0]
    durs = state_df.loc[icorrsts].groupby('idx')['dur'].sum()
    durs = durs.sort_values(ascending=False)
    strdurs = '\n'.join(['{:.1f} ({}%)'.format(1000*dur, int(dur/((t2-t1)/ms)))
                         for dur in durs])
    ncorr, nall = len(durs), len(state_df.idx.unique())
    fres.write('Number of correct states visited: {} / {}'.format(ncorr, nall))
    fres.write('\n\nfor durations (ms):\n\n{}'.format(strdurs))



# %% Make network learn connectivity by examples.

# Generate training examples.


