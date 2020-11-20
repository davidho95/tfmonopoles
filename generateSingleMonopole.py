"""
Generates a single magnetic monopole of charge +1.
"""

import tensorflow as tf
import numpy as np
from theories.GeorgiGlashowSu2Theory import GeorgiGlashowSu2Theory
import FieldTools
import argparse

parser = argparse.ArgumentParser(description="Generate a single monopole")
parser.add_argument("--size", "-s", default=[16], type=int, nargs="*")
parser.add_argument("--vev", "-v", default=1.0, type=float)
parser.add_argument("--gaugeCoupling", "-g", default=1.0, type=float)
parser.add_argument("--selfCoupling", "-l", default=0.5, type=float)
parser.add_argument("--tol", "-t", default=1e-3, type=float)
parser.add_argument("--outputPath", "-o", default=".", type=str)
parser.add_argument("--inputPath", "-i", default="", type=str)
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
    "gaugeCoupling" : args.gaugeCoupling
}


# Set up the initial scalar and gauge fields
inputPath = args.inputPath
if inputPath == "":
    scalarMat, gaugeMat = FieldTools.setMonopoleInitialConditions(
        X, Y, Z, params["vev"]
        )
else:
    scalarMat = np.load(inputPath + "/scalarField.npy")
    gaugeMat = np.load(inputPath + "/gaugeField.npy")

# Convert to tf Variables so gradients can be tracked
scalarField = tf.Variable(scalarMat, trainable=True)
gaugeField = tf.Variable(gaugeMat, trainable=True)

theory = GeorgiGlashowSu2Theory(params)

@tf.function
def lossFn():
    return theory.energy(scalarField, gaugeField)
energy = lossFn()

# Stopping criterion on the maximum value of the gradient
tol = args.tol

# Set up optimiser
opt = tf.keras.optimizers.SGD(
    learning_rate=0.01*args.gaugeCoupling*args.vev, momentum=0.5
    )
numSteps = 0
rmsGrad = 1e6 # Initial value; a big number
maxNumSteps = 10000
printIncrement = 10

while rmsGrad > tol and numSteps < maxNumSteps:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarField, gaugeField]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients so they point in the tangent space 
    # to SU(2)
    grads[1] = FieldTools.projectSu2Gradients(grads[1], gaugeField)

    # Compute max gradient for stopping criterion
    gradSq = FieldTools.innerProduct(grads[0], grads[0], tr=True)
    gradSq += FieldTools.innerProduct(grads[1], grads[1], tr=True, adj=True)

    rmsGrad = tf.sqrt(gradSq)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RMS gradient after " + str(numSteps) + " iterations: " +\
            str(rmsGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
    scalarField.assign(FieldTools.projectToSu2LieAlg(scalarField))
    gaugeField.assign(FieldTools.projectToSu2(gaugeField))

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = args.outputPath
np.save(outputPath + "/X", X.numpy())
np.save(outputPath + "/Y", Y.numpy())
np.save(outputPath + "/Z", Z.numpy())
np.save(outputPath + "/scalarField", scalarField.numpy())
np.save(outputPath + "/gaugeField", gaugeField.numpy())
np.save(outputPath + "/params", params)

