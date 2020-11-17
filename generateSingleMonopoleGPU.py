"""
Generates a single magnetic monopole of charge +1.
"""

import tensorflow as tf
import numpy as np
from GeorgiGlashowSu2Theory import GeorgiGlashowSu2Theory
import FieldTools
from datetime import datetime

# Lattice Size
N = 16

# Set up the lattice
x = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
y = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
z = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)

X,Y,Z = tf.meshgrid(x,y,z, indexing="ij")

# Theory parameters
params = {
    "vev" : 1.0,
    "selfCoupling" : 0.32,
    "gaugeCoupling" : 0.8 
}

# Set up the initial scalar and gauge fields
scalarMat, gaugeMat = \
    FieldTools.setMonopoleInitialConditions(X, Y, Z, params["vev"])

# Convert to tf Variables so gradients can be tracked

# Because of an unresolved issue with the tf-gpu build I am using, the trainable
# variables cannot be complex numbers. As a workaround I'm training the real and
# imaginary parts separately.
scalarMatReal = tf.math.real(scalarMat)
scalarMatImag = tf.math.imag(scalarMat)
gaugeMatReal = tf.math.real(gaugeMat)
gaugeMatImag = tf.math.imag(gaugeMat)
scalarFieldReal = tf.Variable(scalarMatReal, trainable=True)
scalarFieldImag = tf.Variable(scalarMatImag, trainable=True)
gaugeFieldReal = tf.Variable(gaugeMatReal, trainable=True)
gaugeFieldImag = tf.Variable(gaugeMatImag, trainable=True)

myTheory = GeorgiGlashowSu2Theory(params)

@tf.function
def lossFn():
    return myTheory.energy(tf.complex(scalarFieldReal, scalarFieldImag), \
        tf.complex(gaugeFieldReal, gaugeFieldImag))
energy = lossFn()

# Stopping criterion on the maximum value of the gradient
tol = 1e-3

# Set up optimiser
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5)
numSteps = 0
maxGrad = 1e6 # Initial value; a big number
maxNumSteps = 100
printIncrement = 10

startTime = datetime.now()
while maxGrad > tol and numSteps < 100:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [scalarFieldReal, scalarFieldImag, gaugeFieldReal, gaugeFieldImag]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients so they point in the tangent space 
    # to SU(2)
    scalarField = tf.complex(scalarFieldReal, scalarFieldImag)
    gaugeField = tf.complex(gaugeFieldReal, gaugeFieldImag)
    scalarGradsCplx = tf.complex(grads[0], grads[1])
    gaugeGradsCplx = tf.complex(grads[2], grads[3])
    gaugeGradsCplx = FieldTools.projectGaugeGradients(gaugeGradsCplx, gaugeField)
    grads[2] = tf.math.real(gaugeGradsCplx)
    grads[3] = tf.math.imag(gaugeGradsCplx)


    # Compute max gradient for stopping criterion
    maxScalarGrad = tf.math.reduce_max(tf.math.abs(scalarGradsCplx))
    maxGaugeGrad = tf.math.reduce_max(tf.math.abs(gaugeGradsCplx))
    maxGrad = tf.math.reduce_max([maxScalarGrad, maxGaugeGrad])

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("Max gradient after " + str(numSteps) + " iterations: " +\
            str(maxGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
    scalarField = tf.complex(scalarFieldReal, scalarFieldImag)
    gaugeField = tf.complex(gaugeFieldReal, gaugeFieldImag)
    scalarField = (FieldTools.projectToSu2LieAlg(scalarField))
    gaugeField = (FieldTools.projectToSu2(gaugeField))

    scalarFieldReal.assign(tf.math.real(scalarField))
    scalarFieldImag.assign(tf.math.imag(scalarField))
    gaugeFieldReal.assign(tf.math.real(gaugeField))
    gaugeFieldImag.assign(tf.math.imag(gaugeField))

endtime = datetime.now()

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))
print("Time taken: " + str(endtime - startTime))

# Save fields as .npy files for plotting and further analysis
outputPath = "./output/"
np.save(outputPath + "X", X.numpy())
np.save(outputPath + "Y", Y.numpy())
np.save(outputPath + "Z", Z.numpy())
np.save(outputPath + "scalarField", scalarField.numpy())
np.save(outputPath + "gaugeField", gaugeField.numpy())
np.save(outputPath + "params", params)

