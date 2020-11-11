import tensorflow as tf
import numpy as np

def pauliMatrix(cpt):
	if (cpt == 0):
		pauliMat = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
	if (cpt == 1):
		pauliMat = tf.constant([[0, -1j], [1j, 0]], dtype=tf.complex128)
	if (cpt == 2):
		pauliMat = tf.constant([[1, 0], [0, -1]], dtype=tf.complex128)
	return pauliMat

def randomSuLieAlgField(N):
    matReal = tf.random.uniform([N, N, N, 2, 2], dtype=tf.float64)
    matImag = tf.random.uniform([N, N, N, 2, 2], dtype=tf.float64)
    mat = tf.complex(matReal, matImag)
    mat = mat - tf.reshape(0.5*tf.linalg.trace(mat), [N, N, N, 1, 1])*tf.eye(2, batch_shape=[N, N, N], dtype=tf.complex128)
    mat = 0.5*(mat + tf.linalg.adjoint(mat))

    return mat

def vecToSu2LieAlg(inputVectorField):
	inputVectorField = tf.cast(inputVectorField, dtype=tf.complex128)
	inputShape = tf.shape(inputVectorField)[0:3]
	outputShape = tf.concat([inputShape, [2, 2]], 0)
	outputField = tf.zeros(outputShape, dtype=tf.complex128)

	outputField += tf.expand_dims(tf.expand_dims(inputVectorField[:,:,:,0], -1), -1) * pauliMatrix(0)
	outputField += tf.expand_dims(tf.expand_dims(inputVectorField[:,:,:,1], -1), -1) * pauliMatrix(1)
	outputField += tf.expand_dims(tf.expand_dims(inputVectorField[:,:,:,2], -1), -1) * pauliMatrix(2)

	return outputField

def vecToSu2(inputVectorField):
	lieAlgField = vecToSu2LieAlg(inputVectorField)
	return tf.linalg.expm(1j*lieAlgField)

# Sets initial conditions for a single monopole at the origin with twisted boundary conditions.
# X, Y, Z are rank-3 tensors formed as the output of meshgrid; note that 'ij' indexing must be
# used to keep X and Y in the correct order.
def setMonopoleInitialConditions(X, Y, Z, vev):
	latSize = tf.shape(X)
	r = tf.math.sqrt(X**2 + Y**2 + Z**2)

	higgsX = tf.cast(vev, tf.complex128) / np.sqrt(2) * X / r
	higgsY = tf.cast(vev, tf.complex128) / np.sqrt(2) * Y / r
	higgsZ = tf.cast(vev, tf.complex128) / np.sqrt(2) * Z / r

	scalarMat = vecToSu2LieAlg(tf.stack([higgsX, higgsY, higgsZ], -1))

	gaugeVec0 = tf.stack([tf.zeros(latSize, dtype=tf.complex128), Z / r**2, -Y / r**2], -1)
	gaugeVec1 = tf.stack([-Z / r**2, tf.zeros(latSize, dtype=tf.complex128), X / r**2], -1)
	gaugeVec2 = tf.stack([Y / r**2, -X / r**2, tf.zeros(latSize, dtype=tf.complex128)], -1)

	gaugeMat0 = vecToSu2(gaugeVec0)
	gaugeMat1 = vecToSu2(gaugeVec1)
	gaugeMat2 = vecToSu2(gaugeVec2)
	gaugeMat = tf.stack([gaugeMat0, gaugeMat1, gaugeMat2], axis=-3)

	return scalarMat, gaugeMat

def projectToSu2LieAlg(scalarField):
	projectedField = scalarField

	# Make antihermitian
	projectedField.assign(0.5*(projectedField + tf.linalg.adjoint(projectedField)))

	# Make traceless
	trace = tf.linalg.trace(projectedField)
	trace = tf.expand_dims(trace, -1)
	trace = tf.expand_dims(trace, -1)
	projectedField.assign(projectedField - 0.5*trace)

	return projectedField

def projectToSu2(gaugeField):
	projectedField = gaugeField

	# Make proportional to unitary matrix
	determinant = tf.linalg.det(projectedField)
	determinant = tf.expand_dims(determinant, -1)
	determinant = tf.expand_dims(determinant, -1)
	projectedField.assign(0.5*(projectedField + tf.linalg.adjoint(tf.linalg.inv(projectedField)) * determinant))

	# Normalise
	determinant = tf.linalg.det(projectedField)
	determinant = tf.expand_dims(determinant, -1)
	determinant = tf.expand_dims(determinant, -1)
	projectedField.assign(projectedField / tf.math.sqrt(determinant))

	return projectedField

def projectGaugeGradients(gaugeGradients, gaugeField):
	trProduct = tf.linalg.trace(gaugeGradients @ tf.linalg.adjoint(gaugeField))
	trProduct = tf.expand_dims(trProduct, -1)
	trProduct = tf.expand_dims(trProduct, -1)

	# print(tf.shape(trProduct))

	projectedGradients = gaugeGradients - 0.5*trProduct*gaugeField

	return projectedGradients


