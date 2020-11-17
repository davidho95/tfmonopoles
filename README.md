# tfmonopoles
Proof-of-concept for computing monopole and sphaleron field configurations using TensorFlow

## Generate a Magnetic Monopole
The script ```generateSingleMonopole.py``` runs a gradient descent optimisation, minimising the energy of lattice Georgi-Glashow SU(2) Theory with twisted boundary conditions. The minimum-energy field configuration for these boundary conditions is a magnetic monopole of charge +-1.

The optimised scalar and gauge fields, along with the coordinates, are printed to npy files which can be read using ```np.load()``` and used with software such as matplotlib to plot the results.

## Generate an Electroweak Sphaleron
The script ```generateSphaleron.py``` finds a [sphaleron](https://en.wikipedia.org/wiki/Sphaleron) solution to Electroweak theory. It achieves this in two steps:

1. A standard gradient descent optimisation is carried out from suitably chosen initial conditions to take the field as close as possible to the saddle point. No acceleration is used on the gradient descent to exploit the algorithms tendency to get stuck in saddle points.
2. The final convergence is carried out using gradient squared descent: by summing the squares of the gradients of the energy, and using this as the objective function to be minimised. The second-level gradients are normalised on a field by field basis to speed convergence.

The optimised Higgs, isospin and hypercharge fields ar printed to npy files for analysis.

## Notes
These calculations are not suitable for GPU optimisation, as the bulk of the computation is batch multiplication of small (2 x 2) matrices. I find on testing that CPU operations are faster than GPU for all batch sizes that fit in GPU memory.
