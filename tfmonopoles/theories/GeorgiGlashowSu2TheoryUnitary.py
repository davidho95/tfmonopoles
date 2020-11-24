"""
Georgi-Glashow theory restricted to the unitary gauge, with the scalar field
parallel to the third Pauli matrix. Scalar field is now an array of magnitudes,
passed as a complex128 tensor with shape [...,1,1]
"""

import tensorflow as tf
import numpy as np
from tfmonopoles.theories import GeorgiGlashowSu2Theory
from tfmonopoles import FieldTools

class GeorgiGlashowSu2TheoryUnitary(GeorgiGlashowSu2Theory):
    # Inherit most methods from parent, just redefine the ones that change in
    # the unitary gauge

    # Scalar potential
    def scalarPotential(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField)[0:-2], dtype=tf.float64)

        norms = tf.math.real(tf.linalg.trace(scalarField * scalarField))

        energyDensity += self.selfCoupling * (2*norms - self.vev**2)**2
        return energyDensity

    # Gauge covariant derivative
    def covDeriv(self, scalarField, gaugeField, dir):
        scalarFieldShifted = self.shiftScalarField(scalarField, dir, +1)
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
    def shiftScalarField(self, scalarField, dir, sign):
        scalarFieldShifted = tf.roll(scalarField, -sign, dir)

        pauliMatNum = self.boundaryConditions[dir]

        # Only requires flipping if third pauli matrix is used
        if pauliMatNum != 3:
            return scalarFieldShifted

        latShape = tf.shape(scalarField)[0:-2]
        indices = FieldTools.boundaryIndices(latShape, dir, +1)
        updates = -1.0*tf.gather_nd(gaugeFieldShifted, indices)

        scalarFieldShifted = tf.tensor_scatter_nd_update(scalarFieldShifted, indices, updates)

        return scalarFieldShifted
