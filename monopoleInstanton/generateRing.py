"""
Generates a monopole ring from a pair input (in the unitary
gauge). In order to converge to a nontrivial solution, the vev and gauge coupling
must be set so monopoles do not move under gradient flow: vg ~ 1. Optionally, an
external magnetic field may be applied using --externalField.

On small lattices, a nonzero external field is probably required to avoid
converging to the vacuum
"""


import tensorflow as tf
import numpy as np
from tfmonopoles.theories import GeorgiGlashowRadialTheory
from tfmonopoles import FieldTools
import argparse

parser = argparse.ArgumentParser(description="Generate a monopole ring")
parser.add_argument("--vev", "-v", default=1.0, type=float)
parser.add_argument("--gaugeCoupling", "-g", default=1.0, type=float)
parser.add_argument("--selfCoupling", "-l", default=0.5, type=float)
parser.add_argument("--tol", "-t", default=1e-3, type=float)
parser.add_argument("--outputPath", "-o", default="", type=str)
parser.add_argument("--inputPath", "-i", default="", type=str)
parser.add_argument("--numCores", "-n", default=0, type=int)
parser.add_argument("--externalField", "-B", default=0, type=int)

args = parser.parse_args()

if args.numCores != 0:
    tf.config.threading.set_intra_op_parallelism_threads(args.numCores)
    tf.config.threading.set_inter_op_parallelism_threads(args.numCores)

# Load data from input path
inputPath = args.inputPath
inputX = tf.constant(np.load(inputPath + "/X.npy", allow_pickle=True))
inputY = tf.constant(np.load(inputPath + "/Y.npy", allow_pickle=True))
inputZ = tf.constant(np.load(inputPath + "/Z.npy", allow_pickle=True))
inputScalarField = np.load(inputPath + "/scalarField.npy", allow_pickle=True)
inputGaugeField = np.load(inputPath + "/gaugeField.npy", allow_pickle=True)
inputParams = np.load(inputPath + "/params.npy", allow_pickle=True).item()

# Halve the lattice and field size
inputShape = tf.shape(inputX)
R = inputX[int(inputShape[0])//2:,...]
Y = inputY[int(inputShape[0])//2:,...]
Z = inputZ[int(inputShape[0])//2:,...]
latShape = tf.shape(R)

scalarField = inputScalarField[int(inputShape[0])//2:,...]
gaugeField = inputGaugeField[int(inputShape[0])//2:,...]

# Add magnetic field if required
numFluxQuanta = args.externalField
magField = FieldTools.constantMagneticField(R, Y, Z, 0, numFluxQuanta)
gaugeField = FieldTools.linearSuperpose(gaugeField, magField)

# Theory parameters
params = {
    "vev" : args.vev,
    "selfCoupling" : args.selfCoupling,
    "gaugeCoupling" : args.gaugeCoupling,
    "latShape" : latShape
}

theory = GeorgiGlashowRadialTheory(params)

scalarFieldVar = tf.Variable(scalarField)
gaugeFieldVar = tf.Variable(gaugeField)

@tf.function
def lossFn():
    return theory.energy(scalarFieldVar, gaugeFieldVar)
energy = lossFn()

tf.print(energy)

# Stopping criterion on RSS of the gradient
tol = args.tol

# Set up optimiser
opt = tf.keras.optimizers.SGD(
    learning_rate=0.01*args.gaugeCoupling*args.vev, momentum=0.5)
numSteps = 0
rmsGrad = 1e6 # Initial value; a big number
maxNumSteps = 10000
printIncrement = 10

while rmsGrad > tol and numSteps < maxNumSteps:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarFieldVar, gaugeFieldVar]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients so they point in the tangent space 
    # to SU(2)
    grads = theory.processGradients(grads, vars)

    # Compute max gradient for stopping criterion
    gradSq = FieldTools.innerProduct(grads[0], grads[0], tr=True)
    gradSq += FieldTools.innerProduct(grads[1], grads[1], tr=True, adj=True)

    rmsGrad = tf.sqrt(gradSq)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RSS gradient after " + str(numSteps) + " iterations: " +\
            str(rmsGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)
    scalarFieldVar.assign(0.5*(scalarFieldVar + tf.math.conj(scalarFieldVar)))
    gaugeFieldVar.assign(FieldTools.projectToSu2(gaugeFieldVar))

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = args.outputPath
if outputPath != "":
    np.save(outputPath + "/R", R.numpy())
    np.save(outputPath + "/Y", Y.numpy())
    np.save(outputPath + "/Z", Z.numpy())
    np.save(outputPath + "/scalarField", scalarFieldVar.numpy())
    np.save(outputPath + "/gaugeField", gaugeFieldVar.numpy())
    np.save(outputPath + "/params", params)
