import tensorflow as tf
import numpy as np
import GeorgiGlashowSu2Theory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import FieldTools

N = 16

x = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), dtype=tf.complex128)
y = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), dtype=tf.complex128)
z = tf.cast(tf.linspace(-(N-1)/2, (N-1)/2, N), dtype=tf.complex128)

X,Y,Z = tf.meshgrid(x,y,z, indexing='ij')

vev = tf.Variable(1, trainable=False, dtype=tf.float64)
selfCoupling = tf.Variable(0.125, trainable=False, dtype=tf.float64)
gaugeCoupling = tf.Variable(0.5, trainable=False, dtype=tf.float64)

scalarMat, gaugeMat = FieldTools.setMonopoleInitialConditions(X, Y, Z, vev)

scalarField = tf.Variable(scalarMat, trainable=True)
gaugeField = tf.Variable(gaugeMat, trainable=True)


@tf.function
def lossFn():
	return GeorgiGlashowSu2Theory.getEnergy(scalarField, gaugeField, vev, selfCoupling, gaugeCoupling)

energy = lossFn()
print(energy.numpy())
tol = 1e-5

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.3)
numSteps = 0
maxGrad = 1e6

while maxGrad > tol and numSteps < 1000:
	energy = lossFn()

	with tf.GradientTape() as tape:
		loss = lossFn()
	vars = [scalarField, gaugeField]
	grads = tape.gradient(loss, vars)
	grads[1] = FieldTools.projectGaugeGradients(grads[1], gaugeField)
	maxScalarGrad = tf.math.reduce_max(tf.math.abs(grads[0]))
	maxGaugeGrad = tf.math.reduce_max(tf.math.abs(grads[1]))
	maxGrad = tf.math.reduce_max([maxScalarGrad, maxGaugeGrad])

	print(loss.numpy())
	print(maxGrad.numpy())
	opt.apply_gradients(zip(grads, vars))
	numSteps += 1

	scalarField.assign(FieldTools.projectToSu2LieAlg(scalarField))
	gaugeField.assign(FieldTools.projectToSu2(gaugeField))

print("Gradient descent finished in " + str(numSteps) + " iterations")

np.save('X', X.numpy())
np.save('Y', Y.numpy())
np.save('Z', Z.numpy())
np.save('scalarField', scalarField.numpy())
np.save('gaugeField', gaugeField.numpy())

