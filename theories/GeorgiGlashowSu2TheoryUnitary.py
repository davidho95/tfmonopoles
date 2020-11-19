'''
Georgi-Glashow theory restricted to the unitary gauge, with the scalar field
parallel to the third Pauli matrix. Scalar field is now an array of magnitudes,
passed as a complex128 tensor with shape [...,1,1]
'''

import tensorflow as tf
import numpy as np
from theories.GeorgiGlashowSu2Theory import GeorgiGlashowSu2Theory
import FieldTools

class GeorgiGlashowSu2TheoryUnitary(GeorgiGlashowSu2Theory):
    # Inherit most methods from parent, just redefine the ones that change in
    # the unitary gauge

    # Scalar potential
    def scalarPotential(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField)[0:3], dtype=tf.float64)

        norms = tf.math.real(tf.linalg.trace(scalarField * scalarField))

        energyDensity += self.selfCoupling * (2*norms - self.vev**2)**2
        return energyDensity

    # Gauge covariant derivative
    def covDeriv(self, scalarField, gaugeField, dir):
        scalarFieldShifted = self.shiftScalarField(scalarField, dir)
        lieAlgField = scalarField * FieldTools.pauliMatrix(3)
        lieAlgFieldShifted = scalarFieldShifted * FieldTools.pauliMatrix(3)
        covDeriv = gaugeField[:,:,:,dir,:,:] @ lieAlgFieldShifted @\
            tf.linalg.adjoint(gaugeField[:,:,:,dir,:,:]) - lieAlgField
        return covDeriv

    # Projects out abelian subgroup of gauge field
    def getU1Link(self, gaugeField, scalarField, dir):
        u1Link = gaugeField[:,:,:,dir,0,0]
        u1Link = tf.expand_dims(u1Link, -1)
        u1Link = tf.expand_dims(u1Link, -1)

        return u1Link

    # Shifts scalar field using user supplied BC's
    def shiftScalarField(self, scalarField, dir):
        shiftedField = tf.roll(scalarField, -1, dir)

        pauliMatNum = self.boundaryConditions[dir]

        # Only requires flipping if third pauli matrix is used
        if pauliMatNum != 3:
            return shiftedField

        # Create a mask to flip sign at the boundary
        onesBatchShape = list(np.shape(scalarField)[0:3])
        onesBatchShape[dir] -= 1
        onesBatchShape = tf.concat([onesBatchShape, [1,1]], 0)

        minusOnesBatchShape = list(np.shape(scalarField)[0:3])
        minusOnesBatchShape[dir] = 1
        minusOnesBatchShape = tf.concat([minusOnesBatchShape, [1,1]], 0)

        ones = tf.ones(onesBatchShape, dtype=tf.complex128)
        minusOnes = -1.0*tf.ones(minusOnesBatchShape, dtype=tf.complex128)

        boundaryMask = tf.concat([ones, minusOnes], dir)

        shiftedField = boundaryMask * shiftedField

        return shiftedField