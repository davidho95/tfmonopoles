# tfmonopoles
A Python package for computing monopole and sphaleron field configurations using TensorFlow

[![DOI](https://zenodo.org/badge/311992025.svg)](https://zenodo.org/badge/latestdoi/311992025)

## Setup
Currently (this is subject to change in the future) the example scripts `generatePhi4Vacuum.py`, `generateSingleMonpole.py` and `generateSphaleron.py` can be run from the base directory after cloning the repository.

To use `tfmonopoles` theories and functions in other contexts, the package must be installed. The easiest way to do this is to run `pip install .` from the project base directory. To edit the package (e.g. add new theories) while using it, use

```
pip install -e .
```

## Examples
The best way to familiarise oneself with the package is by investigating the example scripts in the main project directory. The following gives an explanation of how to run these scripts out-of-the-box.

### Find the vacuum of Phi<sup>4</sup> theory
The simplest example is the vacuum of a complex scalar field with a quartic potential. Though the vacuum is trivial, the example script demonstrates the minimisation algorithm that forms the key to the more involved examples, using gradient descent to find the global minimum from a random initial field configuration. The script can be run by executing the following:
```
python -u generatePhi4Vacuum.py --size 16 --vev 1 --selfCoupling 0.5 --tol 1e-3 --outputPath [your/output/path]
```
The program outputs the energy and root-sum-square gradient at regular intervals, as well as the final energy of the minimum reached. 

### Generate a Magnetic Monopole
The script ```generateSingleMonopole.py``` runs a gradient descent optimisation, minimising the energy of lattice Georgi-Glashow SU(2) Theory with twisted boundary conditions. The minimum-energy field configuration for these boundary conditions is a magnetic monopole of charge +-1. The script can be run as follows:

```
python -u generateSingleMonopole.py --size 16 --vev 1 --gaugeCoupling 1 --selfCoupling 0.5 --tol 1e-3 --outputPath [your/output/path]
```

where ```[your/output/path]``` is replaced by the directory you want to output to. To play around with the parameters of the theory, simply change the command-line argument values. Depending upon your system, you may need to replace ```python``` with ```python3```.

The optimised scalar and gauge fields, along with the coordinates, are printed to npy files which can be read using ```np.load()``` and used with software such as matplotlib to plot the results.

### Generate an Electroweak Sphaleron
The script `generateSphaleron.py` finds a [sphaleron](https://en.wikipedia.org/wiki/Sphaleron) solution to Electroweak theory. It achieves this in two steps:

1. A standard gradient descent optimisation is carried out from suitably chosen initial conditions to take the field as close as possible to the saddle point. No acceleration is used on the gradient descent to exploit the algorithm's tendency to get stuck in saddle points.
2. The final convergence is carried out using gradient squared descent: by summing the squares of the gradients of the energy, and using this as the objective function to be minimised. The second-level gradients are normalised on a field by field basis to speed convergence.

Like the monopole script, it can be run from the command line:

```
python -u generateSphaleron.py --size 16 --vev 1 --gaugeCoupling 1 --selfCoupling 0.304 --mixingAngle 0.5 --tol 1e-3 --outputPath [your/output/path]
```

The parameters in this example are (roughly) the correct boson mass ratios for the physical Standard Model.

Saddle point finding is much more difficult than minimising, and accordingly this script takes longer to run, but with the given theory parameters it takes around 10 minutes on four intel i5-4460 cores for a 16 x 16 x 16 lattice.

The optimised Higgs, isospin and hypercharge fields are printed to npy files for analysis.

## `tfmonopoles/theories`
The main novelty that `tfmonopoles` provides is a set of classes that define lattice field theories. These can be found in the `tfmonopoles/theories` subdirectory, and can be imported from the `tfmonopoles.theories` module by including, for example
```
from tfmonopoles.theories import Phi4Theory
```
in a Python script. The key element of a theory class is a method mapping a field configuration to a real scalar, bounded from below: in all included theories this is named `energy`. A simple example of an energy method is given in `tfmonopoles/theories/Phi4Theory.py`. This uses finite differences to calculate the kinetic term, and then adds the scalar potential.

New can be added by creating a new class using another theory as a template, or inheriting from a base theory. For example, the class `GeorgiGlashowSu2TheoryUnitary.py` inherits from `GeorgiGlashowSu2Theory.py`, and represents the same physical theory, just with the unitary gauge condition fixed. To import a theory using the above syntax, a relevant line must be added to the `tfmonopoles/theories/__init__.py` file.

## `tfmonopoles/FieldTools.py`
The package also contains a set of functions for common operations on (mainly) SU(2) gauge theories. As the functionality of the package grows, this may be split into submodules.

## Monopole Instantons
The subdirectory `monopoleInstanton` contains the code used to generate monopole instanton solutions in lattice field theory, detailed in [this preprint](https://arxiv.org/abs/2103.12799) For more information [contact the author](mailto:d.ho17@imperial.ac.uk).

## Attribution
If you use this code in research, please cite the following paper, where the code was first used:

```
@article{Ho2021instanton,
    author = "Ho, David L.-J. and Rajantie, Arttu",
    title = "{Instanton solution for Schwinger production of 't Hooft-Polyakov monopoles}",
    eprint = "2103.12799",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "IMPERIAL-TP-2021-DH-04",
    month = "3",
    year = "2021"
}
```

The source code may also be cited directly via its Zenodo DOI:

```
@software{tfmonopoles,
  author       = {Ho, David L.-J.},
  title        = {davidho95/tfmonopoles: First release},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.4972441},
  url          = {https://doi.org/10.5281/zenodo.4972441}
}
```


## Notes
Currently these calculations cannot use GPU optimisation, as tensorflow does have support for complex numbers on GPUs (see relevant issue [here](https://github.com/tensorflow/tensorflow/issues/44834)). A quick way to ensure the program is run on CPU if GPUs are available in a Linux environment is to set the environment variable `CUDA_VISIBLE_DEVICES` using
```
export CUDA_VISIBLE_DEVICES=-1
```

Tested on:
- Python 3.8.5
- TensorFlow 2.3.0
- Windows 10 & Ubuntu 16.04
