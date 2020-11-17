# tfmonopoles
Proof-of-concept for computing monopole and sphaleron field configurations using TensorFlow

## Generate a Magnetic Monopole
The script ```generateSingleMonopole.py``` runs a gradient descent optimisation, minimising the energy of lattice Georgi-Glashow SU(2) Theory with twisted boundary conditions. The minimum-energy field configuration for these boundary conditions is a magnetic monopole of charge +-1. The script can be run as follows:

```python generateSingleMonopole.py --size 16 --vev 1 --gaugeCoupling 1 --selfCoupling 0.5 --tol 1e-3 --outputPath [your_output_path]```

where ```[your_output_path]``` is replaced by the directory you want to output to. To play around with the parameters of the theory, simply change the command-line argument values.

The optimised scalar and gauge fields, along with the coordinates, are printed to npy files which can be read using ```np.load()``` and used with software such as matplotlib to plot the results.

## Generate an Electroweak Sphaleron
The script ```generateSphaleron.py``` finds a [sphaleron](https://en.wikipedia.org/wiki/Sphaleron) solution to Electroweak theory. It achieves this in two steps:

1. A standard gradient descent optimisation is carried out from suitably chosen initial conditions to take the field as close as possible to the saddle point. No acceleration is used on the gradient descent to exploit the algorithm's tendency to get stuck in saddle points.
2. The final convergence is carried out using gradient squared descent: by summing the squares of the gradients of the energy, and using this as the objective function to be minimised. The second-level gradients are normalised on a field by field basis to speed convergence.

Like the monopole script, it can be run from the command line:

```python generateSphaleron.py --size 16 --vev 1 --gaugeCoupling 1 --selfCoupling 0.304 --mixingAngle 0.5 --tol 1e-3 --outputPath [your_output_path]```

The parameters in this example are (roughly) the correct boson mass ratios for the physical Standard Model.

Saddle point finding is much more difficult than minimising, and accordingly this script takes longer to run, but with the given theory parameters it takes around 15 minutes on four intel i5-4460 cores for a 16 x 16 x 16 lattice.

The optimised Higgs, isospin and hypercharge fields are printed to npy files for analysis.

## Notes
These calculations are not suitable for GPU optimisation, as the bulk of the computation is batch multiplication of small (2 x 2) matrices. I find on testing that CPU operations are faster than GPU for all batch sizes that fit in GPU memory.
