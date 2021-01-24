"""
Generates a single magnetic monopole of charge +1.
"""
import tensorflow as tf
import numpy as np
from tfmonopoles.theories import Phi4Theory
import argparse

parser = argparse.ArgumentParser(description="Find the vacuum of Phi^4 Theory")
parser.add_argument("--size", "-s", default=[16], type=int, nargs="*")
parser.add_argument("--vev", "-v", default=1.0, type=float)
parser.add_argument("--selfCoupling", "-l", default=0.5, type=float)
parser.add_argument("--tol", "-t", default=1e-3, type=float)
parser.add_argument("--outputPath", "-o", default="", type=str)
parser.add_argument("--numCores", "-n", default=0, type=int)

args = parser.parse_args()

if args.numCores != 0:
    tf.config.threading.set_intra_op_parallelism_threads(args.numCores)
    tf.config.threading.set_inter_op_parallelism_threads(args.numCores)

# Lattice Size can be a single integer or a list of three; if single integer
# a cubic lattice is generated
latShape = args.size
if len(latShape) == 1:
    Nx = latShape[0]
    Ny = latShape[0]
    Nz = latShape[0]
else:
    Nx, Ny, Nz = latShape

# Set up the lattice
x = tf.cast(tf.linspace(-(Nx-1)/2, (Nx-1)/2, Nx), tf.float64)
y = tf.cast(tf.linspace(-(Ny-1)/2, (Ny-1)/2, Ny), tf.float64)
z = tf.cast(tf.linspace(-(Nz-1)/2, (Nz-1)/2, Nz), tf.float64)

X,Y,Z = tf.meshgrid(x,y,z, indexing="ij")

# Theory parameters
params = {
    "vev" : args.vev,
    "selfCoupling" : args.selfCoupling,
}


# Random complex scalar field
scalarFieldReal = tf.random.uniform([Nx, Ny, Nz], dtype=tf.float64)
scalarFieldImag = tf.random.uniform([Nx, Ny, Nz], dtype=tf.float64)
scalarField = tf.complex(scalarFieldReal, scalarFieldImag)

# Convert to tf Variables so gradients can be tracked
scalarFieldVar = tf.Variable(scalarField, trainable=True)

theory = Phi4Theory(params)

@tf.function
def lossFn():
    return theory.energy(scalarFieldVar)
energy = lossFn()

# Stopping criterion on the maximum value of the gradient
tol = args.tol

# Set up optimiser
# Learning rate is a bit heuristic but works quite well
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-2*args.vev, momentum=0.5
    )
numSteps = 0
rssGrad = 1e6 # Initial value; a big number
maxNumSteps = 10000
printIncrement = 10

while rssGrad > tol and numSteps < maxNumSteps:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarFieldVar]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Compute squared gradient
    gradSq = tf.math.real(tf.math.reduce_sum(grads[0] * tf.math.conj(grads[0])))

    rssGrad = tf.sqrt(gradSq)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RSS gradient after " + str(numSteps) + " iterations: " +\
            str(rssGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = args.outputPath

if outputPath != "":
    np.save(outputPath + "/X", X.numpy())
    np.save(outputPath + "/Y", Y.numpy())
    np.save(outputPath + "/Z", Z.numpy())
    np.save(outputPath + "/scalarField", scalarFieldVar.numpy())
    np.save(outputPath + "/params", params)