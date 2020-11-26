"""
Some tools to manipulate SU(2) gauge and adjoint scalar fields
"""

import tensorflow as tf
import numpy as np

# Generate Pauli matrices, using convention that 0th pauli matrix is the identity
def pauliMatrix(cpt):
    if cpt == 0:
        pauliMat = tf.constant([[1, 0], [0, 1]], dtype=tf.complex128)
    elif cpt == 1:
        pauliMat = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
    elif cpt == 2:
        pauliMat = tf.constant([[0j, -1j], [1j, 0j]], dtype=tf.complex128)
    elif cpt == 3:
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

# Convert an [..., 3] vector field to an [..., 2, 2] field in the SU(2)
# Lie algebra. Equivalent to contraction with a vector field of Pauli matrices 
def vecToSu2LieAlg(inputVectorField):
    inputVectorField = tf.cast(inputVectorField, dtype=tf.complex128)
    inputShape = tf.shape(inputVectorField)[0:-1]
    outputShape = tf.concat([inputShape, [2, 2]], 0)
    outputField = tf.zeros(outputShape, dtype=tf.complex128)

    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[...,0], -1), -1
        ) * pauliMatrix(1)
    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[...,1], -1), -1
        ) * pauliMatrix(2)
    outputField += tf.expand_dims(
        tf.expand_dims(inputVectorField[...,2], -1), -1
        ) * pauliMatrix(3)

    return outputField

# Converts a [..., 3] vector field to a [..., 2, 2] SU(2) field
def vecToSu2(inputVectorField):
    lieAlgField = vecToSu2LieAlg(inputVectorField)
    return tf.linalg.expm(1j*lieAlgField)

# Converts a [..., 2, 2] SU(2) field to a [..., 3] SU(2) field
def su2ToVec(inputField):
    latShape = tf.shape(inputField)[0:-2]
    outputShape = tf.concat([latShape, [3]], 0)

    zeroTol = 1e-15
    cosVecNorm = 0.5*tf.math.real(tf.linalg.trace(inputField))

    outputVec0 = tf.zeros(latShape, dtype=tf.float64)
    outputVec1 = tf.zeros(latShape, dtype=tf.float64)
    outputVec2 = tf.zeros(latShape, dtype=tf.float64)

    vecNorm = tf.math.acos(cosVecNorm)

    # This will clip vec values of +-pi to zero
    outputVec0 = 0.5 * tf.math.divide_no_nan(vecNorm, tf.math.sin(vecNorm)) *\
        tf.math.imag(tf.linalg.trace(inputField @ pauliMatrix(1)))
    outputVec1 = 0.5 * tf.math.divide_no_nan(vecNorm, tf.math.sin(vecNorm)) *\
        tf.math.imag(tf.linalg.trace(inputField @ pauliMatrix(2)))
    outputVec2 = 0.5 * tf.math.divide_no_nan(vecNorm, tf.math.sin(vecNorm)) *\
        tf.math.imag(tf.linalg.trace(inputField @ pauliMatrix(3)))

    return tf.stack([outputVec0, outputVec1, outputVec2], -1)

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
    projectedField = (0.5*(projectedField + \
        tf.linalg.adjoint(projectedField)))

    # Make traceless
    trace = tf.linalg.trace(projectedField)
    trace = tf.expand_dims(trace, -1)
    trace = tf.expand_dims(trace, -1)
    projectedField = (projectedField - 0.5*trace)

    return projectedField

# Project a [..., 2, 2] Matrix field to the SU(2) Lie group.
# This has some array manipulation in to avoid big overheads with calculating
# determinants and inverses for 2 x 2 matrices using built-in functions
def projectToSu2(gaugeField):
    projectedField = gaugeField

    adjugate1 = tf.stack([gaugeField[...,1,1], -gaugeField[...,0,1]], -1)
    adjugate2 = tf.stack([-gaugeField[...,1,0], gaugeField[...,0,0]], -1)

    adjugate = tf.math.conj(tf.stack([adjugate1, adjugate2], -1))

    # Make proportional to unitary matrix
    determinant = gaugeField[...,0,0]*gaugeField[...,1,1] -\
        gaugeField[...,0,1]*gaugeField[...,1,0]
    determinant = tf.expand_dims(determinant, -1)
    determinant = tf.expand_dims(determinant, -1)
    projectedField = (0.5*(projectedField + \
        adjugate))

    # Normalise
    determinant = projectedField[...,0,0]*projectedField[...,1,1] -\
        projectedField[...,0,1]*projectedField[...,1,0]
    determinant = tf.expand_dims(determinant, -1)
    determinant = tf.expand_dims(determinant, -1)
    projectedField = (projectedField / tf.math.sqrt(determinant))

    return projectedField

# Project a [..., 1, 1] field to the U(1) Lie group
def projectToU1(gaugeField):
    projectedField = gaugeField

    # Normalise
    magnitude = tf.abs(gaugeField)
    projectedField = (projectedField / tf.cast(magnitude, tf.complex128))

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

    projectedGradients = u1Gradients - tf.cast(
        tf.math.real(gradFieldProduct), tf.complex128
        ) @ u1Field

    return projectedGradients

# Compute the inner product of two fields
# If trace is True, trace is taken before summing
# If adjoint is true, the first argument is hermitian conjugated
def innerProduct(field1, field2, tr=True, adj=False):
    input1 = tf.linalg.adjoint(field1) if adj else field1
    input2 = field2

    productField = input1 @ input2
    if tr:
        productField = tf.linalg.trace(productField)

    return tf.math.abs(tf.reduce_sum(productField))

# Linearly superpose two SU(2) gauge fields
def linearSuperpose(gaugeField1, gaugeField2):
    # Convert matrices to vectors
    vec1 = su2ToVec(gaugeField1)
    vec2 = su2ToVec(gaugeField2)

    # Add the vectors and output the correspoding SU(2) field
    outputVec = vec1 + vec2

    outputField = vecToSu2(outputVec)

    return outputField

# Generate a constant SU(2) magnetic field of given direction and number of flux
# quanta. Assumes unitary gauge with scalar field parallel to pauli3
def constantMagneticField(X, Y, Z, fieldDir, numFluxQuanta):
    coords = [X,Y,Z]
    latShape = tf.shape(X)
    zeroMat = tf.zeros(latShape, dtype=tf.float64)
    flux = 4*np.pi*tf.cast(numFluxQuanta, tf.float64)

    cpt1 = (fieldDir + 1) % 3
    cpt2 = (fieldDir + 2) % 3 

    gaugeVecDir2 = 0.5*flux / (
        tf.cast(latShape[cpt2]*latShape[cpt1], tf.float64)
            ) * coords[cpt1]

    # Mask for sites on the cpt1 boundary
    cpt1FaceShape = tf.tensor_scatter_nd_update(latShape, [[cpt1]], [1])
    # cpt1FaceShape[cpt1] = 1
    cpt1Mask = tf.ones(cpt1FaceShape, dtype=tf.float64)

    paddings = [[0,0], [0,0], [0,0]]
    paddings[cpt1] = [0, latShape[cpt1] - 1]
    cpt1Mask = tf.pad(cpt1Mask, paddings, constant_values=0)

    gaugeVecDir2 += cpt1Mask*0.5*flux / tf.cast(latShape[cpt2], tf.float64)

    gaugeVecDir1 = zeroMat -\
        0.5*cpt1Mask*coords[cpt2]*flux / tf.cast(latShape[cpt2], tf.float64)

    gaugeCpts = [zeroMat, zeroMat, zeroMat]
    gaugeCpts[fieldDir] = vecToSu2(tf.stack([zeroMat, zeroMat, zeroMat], -1))
    gaugeCpts[cpt1] = vecToSu2(tf.stack([zeroMat, zeroMat, gaugeVecDir1], -1))
    gaugeCpts[cpt2] = vecToSu2(tf.stack([zeroMat, zeroMat, gaugeVecDir2], -1))


    return tf.stack(gaugeCpts, -3)

# Gets the indices of the sites on the boundary
def boundaryIndices(latShape, cpt, sign):
    if sign == +1:
        return sliceIndices(latShape, cpt, latShape[cpt] - 1)
    else:
        return sliceIndices(latShape, cpt, 0)

    return indices

def sliceIndices(latShape, cpt, slicePosition):
    indexVectors = [tf.range(latShape[0]), tf.range(latShape[1]), tf.range(latShape[2])]
    indexVectors[cpt] = slicePosition
    indices = tf.stack(tf.meshgrid(indexVectors[0], indexVectors[1], indexVectors[2], indexing="ij"), -1)

    return indices