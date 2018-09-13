# Sudoku-solver
Solving arbitrary size Sudoku puzzles by Spiking Neural Network

Rules of Sudoku are represented in the connectivity of the network. Input received by network is the initial (incomplete) Sudoku table, and the network finds a solution by settling into a stable (low energy) network state via Winner-Take-All mechanisms.

The process corresponds to models of pattern recognition by completion / perception by incomplete stimuli, where: 

- Solving Sudoku puzzles: forward pass /inference
- Generating Sudoku puzzles: backward pass / spontaneous state settling
- Sampling from multiple solutions compatible with initial puzzle: solving the under-defined perception / pattern recognition problem
- Learning a generative world model connectivity: passing many Sudoku examples through the network with plastic connections

An example puzzle (16 by 16, blue: number given in initial puzzle, green: number correctly found):

<br/>

<img src="https://github.com/davidsamu/Sudoku-solver/blob/master/puzzles/16x16/example/solution.png" width="600">

<br/>

Hardwired connectivity for network solving 4 by 4 puzzles (all connections are inhibitory, each neuron represent a given number in a given cell): 

<br/>

<img src="https://github.com/davidsamu/Sudoku-solver/blob/master/connectivities/size4/hardwired/elev_60_azim_30.png" width="600">

<br/>
