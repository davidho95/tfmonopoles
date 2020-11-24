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
parser.add_argument("--maxSteps", "-M", default=100000, type=int)
parser.add_argument("--momentum", "-p", default=0.95, type=float)

args = parser.parse_args()

if args.numCores != 0:
    tf.config.threading.set_intra_op_parallelism_threads(args.numCores)
    tf.config.threading.set_inter_op_parallelism_threads(args.numCores)

# Load data from input path
inputPath = args.inputPath
R = tf.constant(np.load(inputPath + "/R.npy", allow_pickle=True))
Y = tf.constant(np.load(inputPath + "/Y.npy", allow_pickle=True))
Z = tf.constant(np.load(inputPath + "/Z.npy", allow_pickle=True))
scalarField = np.load(inputPath + "/scalarField.npy", allow_pickle=True)
gaugeField = np.load(inputPath + "/gaugeField.npy", allow_pickle=True)
inputParams = np.load(inputPath + "/params.npy", allow_pickle=True).item()
latShape = tf.shape(R)

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

scalarField = scalarField * tf.cast(params["vev"] / inputParams["vev"], tf.complex128)

theory = GeorgiGlashowRadialTheory(params)

scalarFieldVar = tf.Variable(scalarField)
gaugeFieldVar = tf.Variable(gaugeField)

@tf.function
def lossFn():
    return theory.energy(scalarFieldVar, gaugeFieldVar)
energy = lossFn()

tf.print(energy)

# Stopping criterion on RSS gradient
tol = args.tol

# Just need to satisfy rssGrad < rssGradOld to start the loop
rssGrad = 1e6
rssGradOld = 1e7

numSteps = 0
maxSteps = args.maxSteps
printIncrement = 10
minSteps = 100

# First perform standard gradient descent to get close to the saddle point
opt = tf.keras.optimizers.SGD(learning_rate=0.01*args.gaugeCoupling*args.vev)
while numSteps < minSteps or (rssGrad < rssGradOld and numSteps < maxSteps and rssGrad > tol):
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarFieldVar, gaugeFieldVar]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients
    grads = theory.processGradients(grads, vars)

    # Compute RSS gradient for stopping criterion
    gradSq = FieldTools.innerProduct(grads[0], grads[0], tr=True)
    gradSq += FieldTools.innerProduct(grads[1], grads[1], tr=True, adj=True)

    rssGradOld = rssGrad
    rssGrad = tf.math.sqrt(gradSq)
    # rssGrad = tf.reduce_max(tf.abs(grads[1]))

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RSS gradient after " + str(numSteps) + " iterations: " +\
            str(rssGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields
    scalarFieldVar.assign(0.5*(scalarFieldVar + tf.math.conj(scalarFieldVar)))
    gaugeFieldVar.assign(FieldTools.projectToSu2(gaugeFieldVar))

print("First gradient descent completed in " + str(numSteps) + " iterations")
print("Energy reached: " + str(energy.numpy()))

# Now minimise the RSS gradient summed over all sites
opt = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=args.momentum)
numSteps = 0

while rssGrad > tol and numSteps < maxSteps:
    vars = [scalarFieldVar, gaugeFieldVar]
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as outterTape:
        with tf.GradientTape() as innerTape:
            energy = lossFn()

        # Compute the gradients using automatic differentiation
        grads = innerTape.gradient(energy, vars)

        # Postprocess the gauge field gradients
        grads = theory.processGradients(grads, vars)

        # Compute squared gradients (note that as this is being tracked we can't
        # use the innerProduct function due to passing by value)
        gradSq = tf.math.real(
            tf.reduce_sum(tf.linalg.adjoint(grads[0]) @ grads[0])
            )
        gradSq += tf.math.real(
            tf.reduce_sum(
                tf.linalg.trace(tf.linalg.adjoint(grads[1]) @ grads[1])
                )
            )
        rssGrad = tf.sqrt(gradSq)

    # Compute the second-level gradients (gradient of gradient squared)
    ggrads = outterTape.gradient(gradSq, vars)
    ggrads = theory.processGradients(ggrads, vars)

    # Normalise second-level gradients on a field-by-field basis
    scalarGGradSq = FieldTools.innerProduct(ggrads[0], ggrads[0], adj=True)
    gaugeGGradSq = FieldTools.innerProduct(
        ggrads[1], ggrads[1], tr=True, adj=True
        )

    ggrads[0] /= tf.cast(tf.math.sqrt(scalarGGradSq) + 1e-6, tf.complex128)
    ggrads[1] /= tf.cast(tf.math.sqrt(gaugeGGradSq) + 1e-6, tf.complex128)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RSS gradient after " + str(numSteps) + " iterations: " +\
            str(rssGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(ggrads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
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
