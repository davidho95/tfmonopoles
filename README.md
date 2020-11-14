# tfmonopoles
Proof-of-concept for computing monopole and sphaleron field configurations using TensorFlow

## Generate a Magnetic Monopole
The script ```generateSingleMonopole.py``` runs a gradient descent optimisation, minimising the energy of lattice Georgi-Glashow SU(2) Theory with twisted boundary conditions. The minimum-energy field configuration for these boundary conditions is a magnetic monopole of charge +-1.

The optimised scalar and gauge fields, along with the coordinates, are printed to npy files which can be read using ```np.load()``` and used with software such as matplotlib to plot the results.

## Notes
These calculations are not suitable for GPU optimisation, as the bulk of the computation is batch multiplication of small (2 x 2) matrices. I find on testing that CPU operations are faster than GPU for all batch sizes that fit in GPU memory.
