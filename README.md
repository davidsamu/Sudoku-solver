# Sudoku-solver
Solving arbitrary size Sudoku puzzles by Spiking Neural Network

Rules of Sudoku are represented in the connectivity of the network. Input received by network is the initial (incomplete) Sudoku table, and the network finds a solution by settling into a stable (low energy) network state via Winner-Take-All mechanisms.

The process corresponds to models of pattern recognition by completion / perception by incomplete stimuli, where: 

- Solving Sudoku puzzles: forward pass /inference
- Generating Sudoku puzzles: backward pass / spontaneous state settling
- Sampling from multiple solutions compatible with initial puzzle: solving the under-defined perception / pattern recognition problem
- Learning a generative world model connectivity: passing many Sudoku examples through the network with plastic connections
