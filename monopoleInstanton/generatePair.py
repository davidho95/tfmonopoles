"""
Generates a pair of monopoles from a single monopole input (in the unitary
gauge). In order to converge to a nontrivial solution, the vev and gauge coupling
must be set so monopoles do not move under gradient flow: vg ~ 1. Optionally, an
external magnetic field may be applied using --externalField.
"""

import tensorflow as tf
import numpy as np
from tfmonopoles.theories import GeorgiGlashowSu2TheoryUnitary
from tfmonopoles import FieldTools
import argparse

parser = argparse.ArgumentParser(description="Generate a monopole-antimonopole pair")
parser.add_argument("--vev", "-v", default=1.0, type=float)
parser.add_argument("--gaugeCoupling", "-g", default=1.0, type=float)
parser.add_argument("--selfCoupling", "-l", default=0.5, type=float)
parser.add_argument("--tol", "-t", default=1e-3, type=float)
parser.add_argument("--outputPath", "-o", default="", type=str)
parser.add_argument("--inputPath", "-i", default="", type=str)
parser.add_argument("--numCores", "-n", default=0, type=int)
parser.add_argument("--separation", "-d", default=8, type=int)
parser.add_argument("--externalField", "-B", default=0, type=int)

args = parser.parse_args()

if args.numCores != 0:
    tf.config.threading.set_intra_op_parallelism_threads(args.numCores)
    tf.config.threading.set_inter_op_parallelism_threads(args.numCores)

# Load data from input path
inputPath = args.inputPath
X = tf.constant(np.load(inputPath + "/X.npy", allow_pickle=True))
Y = tf.constant(np.load(inputPath + "/Y.npy", allow_pickle=True))
Z = tf.constant(np.load(inputPath + "/Z.npy", allow_pickle=True))
inputScalarField = np.load(inputPath + "/scalarField.npy", allow_pickle=True)
inputGaugeField = np.load(inputPath + "/gaugeField.npy", allow_pickle=True)
inputParams = np.load(inputPath + "/params.npy", allow_pickle=True).item()

# Infer lattice size from input
latShape = tf.shape(inputScalarField)

# Theory to work on the single pole field
singlePoleTheory = GeorgiGlashowSu2TheoryUnitary(inputParams)

# Find monopole position by looking for nonzero values of divB
magX = singlePoleTheory.magneticField(inputGaugeField, inputScalarField, 0)
magY = singlePoleTheory.magneticField(inputGaugeField, inputScalarField, 1)
magZ = singlePoleTheory.magneticField(inputGaugeField, inputScalarField, 2)
magXShifted = tf.Variable(tf.roll(magX,-1,0))
magXShifted[-1,:,:].assign(-magXShifted[-1,:,:])
magYShifted = tf.Variable(tf.roll(magY,-1,1))
magZShifted = tf.Variable(tf.roll(magZ,-1,2))
divB =  magXShifted + magYShifted + magZShifted - magX - magY - magZ
monopoleCoords = tf.cast(tf.where(tf.greater(tf.math.abs(divB), 1e-3)), tf.int32)
monopoleXCoord = monopoleCoords[0][0]

print(monopoleXCoord)

# Calculate shifts required to give the desired separation, centred about the
# origin
separation = args.separation
desiredLeftPos = latShape[0] // 2 - (separation + 1) // 2
desiredRightPos = latShape[0] // 2 + separation // 2
shiftNumLeft = ((desiredRightPos - monopoleXCoord) + latShape[0]) % latShape[0]
shiftNumRight = ((desiredLeftPos - monopoleXCoord) + latShape[0]) % latShape[0]

# Shift the poles around, obeying the BCs
leftGaugeField = inputGaugeField
leftScalarField = inputScalarField
rightGaugeField = inputGaugeField
rightScalarField = inputScalarField
for ii in range(shiftNumLeft):
    leftGaugeField = singlePoleTheory.shiftGaugeField(leftGaugeField, 0, +1)
    leftScalarField = singlePoleTheory.shiftScalarField(leftScalarField, 0, +1)

# Because shiftNumRight > monopoleXCoord, the monopole ends up conjugated
for ii in range(shiftNumRight):
    rightGaugeField = singlePoleTheory.shiftGaugeField(rightGaugeField, 0, +1)
    rightScalarField = singlePoleTheory.shiftScalarField(rightScalarField, 0, +1)

gaugeField = FieldTools.linearSuperpose(leftGaugeField, rightGaugeField)
scalarField = 0.5*(leftScalarField + rightScalarField)

# Shift once more to centre the pair
gaugeField = tf.roll(gaugeField, -1, 0)
scalarField = tf.roll(scalarField, -1, 0)

numFluxQuanta = args.externalField
# Negative flux quanta results in lowering of energy (dipole points along
# negative x axis)
magField = FieldTools.constantMagneticField(X, Y, Z, 0, -numFluxQuanta)
gaugeField = FieldTools.linearSuperpose(gaugeField, magField)

gaugeFieldVar = tf.Variable(gaugeField, trainable=True)
scalarFieldVar = tf.Variable(scalarField, trainable=True)

params = params = {
    "vev" : args.vev,
    "selfCoupling" : args.selfCoupling,
    "gaugeCoupling" : args.gaugeCoupling,
    "boundaryConditions" : [0, 0, 0]
}

theory = GeorgiGlashowSu2TheoryUnitary(params)

@tf.function
def lossFn():
    return theory.energy(scalarFieldVar, gaugeFieldVar)
energy = lossFn()

# Stopping criterion on the maximum value of the gradient
tol = args.tol

# Set up optimiser
opt = tf.keras.optimizers.SGD(
    learning_rate=0.02*args.gaugeCoupling*args.vev, momentum=0.9, nesterov=True
    )
numSteps = 0
rmsGrad = 1e6 # Initial value; a big number
maxNumSteps = 100000
minSteps = 1000 # Gets stuck in an unwanted saddle point without these
printIncrement = 10

while numSteps < minSteps or (rmsGrad > tol and numSteps < maxNumSteps):
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarFieldVar, gaugeFieldVar]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients so they point in the tangent space 
    # to SU(2)
    grads[1] = FieldTools.projectSu2Gradients(grads[1], gaugeFieldVar)

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

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
    scalarFieldVar.assign(0.5*(scalarFieldVar + tf.math.conj(scalarFieldVar)))
    gaugeFieldVar.assign(FieldTools.projectToSu2(gaugeFieldVar))

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = args.outputPath
if outputPath != "":
    np.save(outputPath + "/X", X.numpy())
    np.save(outputPath + "/Y", Y.numpy())
    np.save(outputPath + "/Z", Z.numpy())
    np.save(outputPath + "/scalarField", scalarFieldVar.numpy())
    np.save(outputPath + "/gaugeField", gaugeFieldVar.numpy())
    np.save(outputPath + "/params", params)
