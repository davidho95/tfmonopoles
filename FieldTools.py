'''
Some tools to manipulate SU(2) gauge and adjoint scalar fields
'''

import tensorflow as tf
import numpy as np

# Generate Pauli matrices
def pauliMatrix(cpt):
    if (cpt == 0):
        pauliMat = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
    if (cpt == 1):
        pauliMat = tf.constant([[0j, -1j], [1j, 0j]], dtype=tf.complex128)
    if (cpt == 2):
        pauliMat = tf.constant([[1, 0], [0, -1]], dtype=tf.complex128)
    return pauliMat

# Generate an [N, N, N] field taking random values in the SU(2) Lie algebra.
def randomSu2LieAlgField(N):
    matReal = tf.random.uniform([N, N, N, 2, 2], dtype=tf.float64)
    matImag = tf.random.uniform([N, N, N, 2, 2], dtype=tf.float64)
    mat = tf.complex(matReal, matImag)

    trace = tf.linalg.trace(mat)
    trace = tf.expand_dims(trace, -1)
    trace = tf.expand_dims(trace, -1)
    identity = tf.eye(2, batch_shape=[N, N, N], dtype=tf.complex128)
    mat = mat - trace*identity
    mat = 0.5*(mat + tf.linalg.adjoint(mat))

    return mat

# Convert an [N, N, N, 3] vector field to an [N, N, N, 2, 2] field in the SU(2)
# Lie algebra. Equivalent to contraction with a vector field of Pauli matrices 
def vecToSu2LieAlg(inputVectorField):
    inputVectorField = tf.cast(inputVectorField, dtype=tf.complex128)
    inputShape = tf.shape(inputVectorField)[0:3]
    outputShape = tf.concat([inputShape, [2, 2]], 0)
    outputField = tf.zeros(outputShape, dtype=tf.complex128)

    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[:,:,:,0], -1), -1
        ) * pauliMatrix(0)
    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[:,:,:,1], -1), -1
        ) * pauliMatrix(1)
    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[:,:,:,2], -1), -1
        ) * pauliMatrix(2)

    return outputField

def vecToSu2(inputVectorField):
    lieAlgField = vecToSu2LieAlg(inputVectorField)
    return tf.linalg.expm(1j*lieAlgField)

# Sets initial conditions for a single monopole at the origin with twisted
# boundary conditions. X, Y, Z are rank-3 tensors formed as the output of 
# meshgrid; note that 'ij' indexing must be used to keep X and Y in the correct
# order.
def setMonopoleInitialConditions(X, Y, Z, vev):
    latSize = tf.shape(X)
    r = tf.math.sqrt(X**2 + Y**2 + Z**2)

    higgsX = vev / np.sqrt(2) * X / r
    higgsY = vev / np.sqrt(2) * Y / r
    higgsZ = vev / np.sqrt(2) * Z / r

    scalarMat = vecToSu2LieAlg(tf.stack([higgsX, higgsY, higgsZ], -1))

    zeroMat = tf.zeros(latSize, dtype=tf.float64)
    gaugeVec0 = tf.stack([zeroMat, Z / r**2, -Y / r**2], -1)
    gaugeVec1 = tf.stack([-Z / r**2, zeroMat, X / r**2], -1)
    gaugeVec2 = tf.stack([Y / r**2, -X / r**2, zeroMat], -1)

    gaugeMat0 = vecToSu2(gaugeVec0)
    gaugeMat1 = vecToSu2(gaugeVec1)
    gaugeMat2 = vecToSu2(gaugeVec2)
    gaugeMat = tf.stack([gaugeMat0, gaugeMat1, gaugeMat2], axis=-3)

    return scalarMat, gaugeMat

# Sets initial conditions for an Electroweak sphaleron at the origin with periodic
# boundary conditions. X, Y, Z are rank-3 tensors formed as the output of 
# meshgrid; note that 'ij' indexing must be used to keep X and Y in the correct
# order.
def setSphaleronInitialConditions(X, Y, Z, vev, gaugeCouping):
    latSize = tf.shape(X)
    r = tf.math.sqrt(X**2 + Y**2 + Z**2)

    zeroMat = tf.zeros(latSize, dtype=tf.float64)
    gaugeFn = 1/tf.math.cosh(vev*gaugeCouping*r/3)
    higgsFn = tf.cast(tf.math.tanh(vev*gaugeCouping*r/3), tf.complex128)

    higgsMat = 1/np.sqrt(2) * vev * tf.ones(latSize, dtype=tf.complex128) * higgsFn
    higgsMat = tf.expand_dims(higgsMat, -1)
    higgsMat = tf.expand_dims(higgsMat, -1)

    isospinVecX = tf.stack([zeroMat, Z / r**2 * gaugeFn, -Y / r**2 * gaugeFn], -1)
    isospinVecY = tf.stack([-Z / r**2 * gaugeFn, zeroMat, X / r**2 * gaugeFn], -1)
    isospinVecZ = tf.stack([Y / r**2 * gaugeFn, -X / r**2 * gaugeFn, zeroMat], -1)

    isospinMatX = vecToSu2(isospinVecX)
    isospinMatY = vecToSu2(isospinVecY)
    isospinMatZ = vecToSu2(isospinVecZ)
    isospinMat = tf.stack([isospinMatX, isospinMatY, isospinMatZ], axis=-3)

    hyperchargeMat = tf.ones(latSize, dtype=tf.complex128)
    hyperchargeMat = tf.expand_dims(hyperchargeMat, -1)
    hyperchargeMat = tf.expand_dims(hyperchargeMat, -1)

    hyperchargeMat = tf.stack([hyperchargeMat, hyperchargeMat, hyperchargeMat], -3)

    return higgsMat, isospinMat, hyperchargeMat


# Project a [..., 2, 2] Matrix field to the SU(2) Lie algebra.
def projectToSu2LieAlg(scalarField):
    projectedField = scalarField

    # Make antihermitian
    projectedField.assign(0.5*(projectedField + \
        tf.linalg.adjoint(projectedField)))

    # Make traceless
    trace = tf.linalg.trace(projectedField)
    trace = tf.expand_dims(trace, -1)
    trace = tf.expand_dims(trace, -1)
    projectedField.assign(projectedField - 0.5*trace)

    return projectedField

# Project a [..., 2, 2] Matrix field to the SU(2) Lie group.
def projectToSu2(gaugeField):
    projectedField = gaugeField

    # Make proportional to unitary matrix
    determinant = tf.linalg.det(projectedField)
    determinant = tf.expand_dims(determinant, -1)
    determinant = tf.expand_dims(determinant, -1)
    projectedField.assign(0.5*(projectedField + \
        tf.linalg.adjoint(tf.linalg.inv(projectedField)) * determinant))

    # Normalise
    determinant = tf.linalg.det(projectedField)
    determinant = tf.expand_dims(determinant, -1)
    determinant = tf.expand_dims(determinant, -1)
    projectedField.assign(projectedField / tf.math.sqrt(determinant))

    return projectedField

# Project a [..., 1, 1] field to the U(1) Lie group
def projectToU1(gaugeField):
    projectedField = gaugeField

    # Normalise
    magnitude = tf.abs(gaugeField)
    projectedField.assign(projectedField / tf.cast(magnitude, tf.complex128))

    return projectedField

# Remove the part of a gradient field that points away from the SU(2) manifold
def projectSu2Gradients(su2Gradients, su2Field):
    trProduct = tf.linalg.trace(su2Gradients @ tf.linalg.adjoint(su2Field))
    trProduct = tf.expand_dims(trProduct, -1)
    trProduct = tf.expand_dims(trProduct, -1)

    # print(tf.shape(trProduct))

    projectedGradients = su2Gradients - 0.5*trProduct*su2Field

    return projectedGradients


# # Remove the part of a gradient field that points away from the U(1) manifold
def projectU1Gradients(u1Gradients, u1Field):
    gradFieldProduct = u1Gradients @ tf.linalg.adjoint(u1Field)

    # print(tf.shape(trProduct))

    projectedGradients = u1Gradients - tf.cast(tf.math.real(gradFieldProduct), tf.complex128) @ u1Field

    return projectedGradients