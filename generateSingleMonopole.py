"""
Generates a single magnetic monopole of charge +-1.
"""

import tensorflow as tf
import numpy as np
import GeorgiGlashowSu2Theory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import FieldTools

# Lattice Size
N = 16

# Set up the lattice
x = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
y = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
z = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)

X,Y,Z = tf.meshgrid(x,y,z, indexing="ij")

# Theory parameters
vev = tf.Variable(1, trainable=False, dtype=tf.float64)
selfCoupling = tf.Variable(0.32, trainable=False, dtype=tf.float64)
gaugeCoupling = tf.Variable(0.8, trainable=False, dtype=tf.float64)

# Set up the initial scalar and gauge fields
scalarMat, gaugeMat = FieldTools.setMonopoleInitialConditions(X, Y, Z, vev)

# Convert to tf Variables so gradients can be tracked
scalarField = tf.Variable(scalarMat, trainable=True)
gaugeField = tf.Variable(gaugeMat, trainable=True)

@tf.function
def lossFn():
    return GeorgiGlashowSu2Theory.getEnergy(scalarField, gaugeField, vev, \
        selfCoupling, gaugeCoupling)

energy = lossFn()
print("Initial energy: " + str(energy.numpy()))

# Stopping criterion on the maximum value of the gradient
tol = 1e-6

# Set up optimiser
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
numSteps = 0
maxGrad = 1e6 # Initial value; a big number
maxNumSteps = 10000

while maxGrad > tol and numSteps < maxNumSteps:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarField, gaugeField]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients so they point in the
    # tangent space to SU(2)
    grads[1] = FieldTools.projectGaugeGradients(grads[1], gaugeField)

    # Compute max gradient for stopping criterion
    maxScalarGrad = tf.math.reduce_max(tf.math.abs(grads[0]))
    maxGaugeGrad = tf.math.reduce_max(tf.math.abs(grads[1]))
    maxGrad = tf.math.reduce_max([maxScalarGrad, maxGaugeGrad])

    print(energy.numpy())
    print(maxGrad.numpy())

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
    scalarField.assign(FieldTools.projectToSu2LieAlg(scalarField))
    gaugeField.assign(FieldTools.projectToSu2(gaugeField))

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = "./output/"
np.save(outputPath + "X", X.numpy())
np.save(outputPath + "Y", Y.numpy())
np.save(outputPath + "Z", Z.numpy())
np.save(outputPath + "scalarField", scalarField.numpy())
np.save(outputPath + "gaugeField", gaugeField.numpy())

