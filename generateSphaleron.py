import tensorflow as tf
import numpy as np
from ElectroweakTheoryUnitary import ElectroweakTheory
import FieldTools

# Lattic size
N = 16

# Theory parameters
params = {
    "vev" : 1,
    "gaugeCoupling" : 1, # $\lambda / g^2$ reflects physical boson mass ratio
    "selfCoupling" : 0.304, 
    "mixingAngle" : 0.5 # Close to the physical value
}

# Set up the lattice
x = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
y = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)
z = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), tf.float64)

X,Y,Z = tf.meshgrid(x,y,z, indexing="ij")

theory = ElectroweakTheory(params)

higgsMat, isospinMat, hyperchargeMat = FieldTools.setSphaleronInitialConditions(
    X, Y, Z, params["vev"], params["gaugeCoupling"]
    )

# Set up variables so tf can watch the gradients
higgsField = tf.Variable(higgsMat, trainable=True)
isospinField = tf.Variable(isospinMat, trainable=True)
hyperchargeField = tf.Variable(hyperchargeMat, trainable=True)

@tf.function
def lossFn():
    return theory.energy(higgsField, isospinField, hyperchargeField)
energy = lossFn()

# Stopping criterion on RMS gradient
tol = 1e-3

# Just need to satisfy rmsGrad < rmsGradOld to start the loop
rmsGrad = 1e6
rmsGradOld = 1e7

numSteps = 0
maxNumSteps = 1000000
printIncrement = 10

# First perform standard gradient descent to get close to the saddle point
opt = tf.keras.optimizers.SGD(learning_rate=0.02)
while rmsGrad < rmsGradOld and numSteps < maxNumSteps:
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as tape:
        energy = lossFn()

    vars = [higgsField, isospinField, hyperchargeField]

    # Compute the gradients using automatic differentiation
    grads = tape.gradient(energy, vars)

    # Postprocess the gauge field gradients
    grads[1] = FieldTools.projectSu2Gradients(grads[1], isospinField)
    grads[2] = FieldTools.projectU1Gradients(grads[2], hyperchargeField)

    # Compute rms gradient for stopping criterion
    gradSq = FieldTools.innerProduct(grads[0], grads[0], adj=True)
    gradSq += FieldTools.innerProduct(grads[1], grads[1], tr=True, adj=True)
    gradSq += FieldTools.innerProduct(grads[1], grads[1], adj=True)

    rmsGradOld = rmsGrad
    rmsGrad = tf.math.sqrt(gradSq)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RMS gradient after " + str(numSteps) + " iterations: " +\
            str(rmsGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(grads, vars))
    numSteps += 1

    # Postprocess the fields
    higgsField.assign(0.5 * (higgsField + tf.math.conj(higgsField)))
    isospinField.assign(FieldTools.projectToSu2(isospinField))
    hyperchargeField.assign(FieldTools.projectToU1(hyperchargeField))

print("First gradient descent completed in " + str(numSteps) + " iterations")
print("Energy reached: " + str(energy.numpy()))

# Now minimise the RMS gradient summed over all sites
opt = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
numSteps = 0

while rmsGrad > tol and numSteps < maxNumSteps:
    vars = [higgsField, isospinField, hyperchargeField]
    # Compute the field energy, with tf watching the variables
    with tf.GradientTape() as outterTape:
        with tf.GradientTape() as innerTape:
            energy = lossFn()

        # Compute the gradients using automatic differentiation
        grads = innerTape.gradient(energy, vars)

        # Postprocess the gauge field gradients
        grads[1] = FieldTools.projectSu2Gradients(grads[1], isospinField)
        grads[2] = FieldTools.projectU1Gradients(grads[2], hyperchargeField)

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
        gradSq += tf.math.real(
            tf.reduce_sum(tf.linalg.adjoint(grads[2]) @ grads[2])
            )
        rmsGrad = tf.sqrt(gradSq)

    # Compute the second-level gradients (gradient of gradient squared)
    ggrads = outterTape.gradient(gradSq, vars)
    ggrads[1] = FieldTools.projectSu2Gradients(ggrads[1], isospinField)
    ggrads[2] = FieldTools.projectU1Gradients(ggrads[2], hyperchargeField)

    # Normalise second-level gradients on a field-by-field basis
    higgsGGradSq = FieldTools.innerProduct(ggrads[0], ggrads[0], adj=True)
    isospinGGradSq = FieldTools.innerProduct(
        ggrads[1], ggrads[1], tr=True, adj=True
        )
    hyperchargeGGradSq = FieldTools.innerProduct(ggrads[2], ggrads[2], adj=True)

    ggrads[0] /= tf.cast(tf.math.sqrt(higgsGGradSq) + 1e-6, tf.complex128)
    ggrads[1] /= tf.cast(tf.math.sqrt(isospinGGradSq) + 1e-6, tf.complex128)
    ggrads[2] /= tf.cast(tf.math.sqrt(hyperchargeGGradSq) + 1e-6, tf.complex128)

    if (numSteps % printIncrement == 0):
        print("Energy after " + str(numSteps) + " iterations:       " +\
            str(energy.numpy()))
        print("RMS gradient after " + str(numSteps) + " iterations: " +\
            str(rmsGrad.numpy()))

    # Perform the gradient descent step
    opt.apply_gradients(zip(ggrads, vars))
    numSteps += 1

    # Postprocess the fields to avoid drift away from SU(2)/its Lie algebra
    higgsField.assign(0.5 * (higgsField + tf.math.conj(higgsField)))
    isospinField.assign(FieldTools.projectToSu2(isospinField))
    hyperchargeField.assign(FieldTools.projectToU1(hyperchargeField))

print("Gradient descent finished in " + str(numSteps) + " iterations")
print("Final energy: " + str(energy.numpy()))

# Save fields as .npy files for plotting and further analysis
outputPath = "./output/"
np.save(outputPath + "X", X.numpy())
np.save(outputPath + "Y", Y.numpy())
np.save(outputPath + "Z", Z.numpy())
np.save(outputPath + "higgsField", higgsField.numpy())
np.save(outputPath + "isospinField", isospinField.numpy())
np.save(outputPath + "hyperchargeField", hyperchargeField.numpy())
np.save(outputPath + "params", params)