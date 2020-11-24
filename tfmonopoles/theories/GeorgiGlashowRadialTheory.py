"""
Class for calculating field properties in 4d Georgi-Glashow Su(2) Theory
compactified in the x-t angular variable. The unitary gauge is used.
"""

import tensorflow as tf
import numpy as np
from tfmonopoles.theories import GeorgiGlashowSu2TheoryUnitary
from tfmonopoles import FieldTools

class GeorgiGlashowRadialTheory(GeorgiGlashowSu2TheoryUnitary):
    # Params is a dictionary with keys "vev", "selfCoupling" and "gaugeCoupling"
    def __init__(self, params):
        super().__init__(params)
        self.latShape = tf.constant(params["latShape"])

        r = tf.linspace(tf.cast(0.5, tf.float64), tf.cast(self.latShape[0], tf.float64) - 0.5, self.latShape[0])
        y = tf.linspace(-tf.cast(self.latShape[1] - 1, tf.float64) / 2, tf.cast(self.latShape[1] - 1, tf.float64) / 2, self.latShape[1]) 
        z = tf.linspace(-tf.cast(self.latShape[2] - 1, tf.float64) / 2, tf.cast(self.latShape[2] - 1, tf.float64) / 2, self.latShape[2])

        self.metric, _, _ = tf.meshgrid(r, y, z, indexing="ij")

    # This is called energy despite being the action for consistency with other
    # 3d theories.
    # scalarField is a [latSize, 1, 1] complex128 tensor, but only the real
    # part is used
    # gaugeField is a [latSize, 3, 2, 2] complex128 tensor taking values in SU(2)
    def energy(self, scalarField, gaugeField):
        return tf.math.reduce_sum(self.energyDensity(scalarField, gaugeField))

    def energyDensity(self, scalarField, gaugeField):
        energyDensity = tf.zeros(self.latShape, dtype=tf.float64)

        energyDensity += self.ymTerm(gaugeField)
        energyDensity += self.covDerivTerm(scalarField, gaugeField)
        energyDensity += self.scalarPotential(scalarField)

        # Integrate over the z-t angular variable
        return 2*np.pi*energyDensity

    # Wilson action
    def ymTerm(self, gaugeField):
        energyDensity = tf.zeros(self.latShape, dtype=tf.float64)

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 2/self.gaugeCoupling**2 * tf.math.real((2 - \
                    tf.linalg.trace(self.avgPlaquette(gaugeField, ii, jj)))) *\
                    self.metric

        return energyDensity

    # Gauge kinetic term for the scalar field
    def covDerivTerm(self, scalarField, gaugeField):
        energyDensity = tf.zeros(self.latShape, dtype=tf.float64)
        numDims = 3
        for ii in range(numDims):
            avgCovDerivSq = self.avgCovDerivSq(scalarField, gaugeField, ii)
            energyDensity += tf.math.real(tf.linalg.trace(avgCovDerivSq)) *\
                self.metric

        return energyDensity

    # Scalar potential
    def scalarPotential(self, scalarField):
        energyDensity = tf.zeros(self.latShape, dtype=tf.float64)

        norms = tf.math.real(tf.linalg.trace(scalarField * scalarField))

        energyDensity += self.selfCoupling * (2*norms - self.vev**2)**2 * self.metric
        return energyDensity

    # Averaged plaquette that lies on a lattice site
    def avgPlaquette(self, gaugeField, dir1, dir2):
        plaquette = self.plaquette(gaugeField, dir1, dir2)
        avgPlaquette = plaquette
        avgPlaquette += self.shiftPlaquette(plaquette, dir1, dir2, dir1, -1)
        avgPlaquette += self.shiftPlaquette(plaquette, dir1, dir2, dir2, -1)
        avgPlaquette += self.shiftPlaquette(
            self.shiftPlaquette(plaquette, dir1, dir2, dir1, -1), dir1, dir2,
            dir2, -1)

        return 0.25*avgPlaquette

    # Average squared covariant derivative that lies on a lattice site. Note
    # we cannot average the covariant derivative and then square as this would
    # result in nonlocality
    def avgCovDerivSq(self, scalarField, gaugeField, dir):
        covDeriv = self.covDeriv(scalarField, gaugeField, dir)
        covDerivSq = covDeriv @ covDeriv
        covDerivSqShiftedBwd = self.shiftCovDeriv(covDerivSq, dir, -1)
        avgCovDerivSq = covDerivSq + covDerivSqShiftedBwd

        return 0.5*avgCovDerivSq

    # Shifts scalar field using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards. As there is a physical boundary, shifting in one
    # direction then another is not an identity operation.
    def shiftScalarField(self, scalarField, dir, sign):
        # Moving one site forwards is equivalent to shifting the whole field
        # backwards, hence the minus sign (active/passive transform)
        shiftedField = tf.roll(scalarField, -sign, dir)

        if dir != 0:
            return shiftedField

        # Reflecting boundary conditions in the r direction: replace boundary
        # values with the corresponding values from the unshifted field
        boundaryMask = self.scalarBoundaryMask(sign)
        shiftedField = shiftedField - shiftedField * boundaryMask
        shiftedField = shiftedField + scalarField * boundaryMask

        return shiftedField

    # Shifts gauge field using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards. As there is a physical boundary, shifting in one
    # direction then another is not an identity operation.
    def shiftGaugeField(self, gaugeField, dir, sign):
        # Moving one site forwards is equivalent to shifting the whole field
        # backwards, hence the minus sign (active/passive transform)
        shiftedField = tf.roll(gaugeField, -sign, dir)

        if dir != 0:
            return shiftedField

        # Reflecting boundary conditions in the r direction: replace boundary
        # values with the corresponding values from the unshifted field
        boundaryMask = self.gaugeBoundaryMask(sign)
        shiftedField = shiftedField - shiftedField @ boundaryMask
        shiftedField = shiftedField + gaugeField @ boundaryMask

        if sign == -1:
        # Set the r-links at the origin to the identity
            rBoundaryMask = self.rGaugeBoundaryMask()
            shiftedField = shiftedField - shiftedField @ rBoundaryMask
            shiftedField = shiftedField + rBoundaryMask

        return shiftedField

    # Shifts plaquette using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards 
    def shiftPlaquette(self, plaquette, plaqDir1, plaqDir2, shiftDir, sign):
        # Moving one site forwards is equivalent to shifting the whole field
        # backwards, hence the minus sign (active/passive transform)
        shiftedField = tf.roll(plaquette, -sign, shiftDir)

        if shiftDir != 0:
            return shiftedField

        # Reflecting boundary conditions in the r direction: replace boundary
        # values with the corresponding values from the unshifted field
        boundaryMask = self.plaquetteBoundaryMask(sign)
        shiftedField = shiftedField - shiftedField @ boundaryMask
        shiftedField = shiftedField + plaquette @ boundaryMask

        if sign == -1 and (plaqDir1 == 0 or plaqDir2 == 0):
        # Set the plaquettes straddling the origin to the identity
            rBoundaryMask = self.plaquetteBoundaryMask(sign)
            shiftedField = shiftedField - shiftedField @ rBoundaryMask
            shiftedField = shiftedField + rBoundaryMask

        return shiftedField

    # Shifts covariant derivative using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards 
    def shiftCovDeriv(self, covDeriv, dir, sign):
        shiftedField = tf.roll(covDeriv, -sign, dir)

        if dir != 0:
            return shiftedField

        boundaryMask = self.scalarBoundaryMask(sign)
        shiftedField = shiftedField - shiftedField * boundaryMask
        if sign == +1:
            shiftedField = shiftedField + covDeriv * boundaryMask

        return shiftedField

    def scalarBoundaryMask(self, sign):
        zeroBatchShape = self.latShape
        zeroBatchShape = tf.tensor_scatter_nd_update(zeroBatchShape, [[0]], [zeroBatchShape[0] - 1])
        zeroBatchShape = tf.concat([zeroBatchShape, [1,1]], 0)
        
        onesBatchShape = self.latShape
        onesBatchShape = tf.tensor_scatter_nd_update(onesBatchShape, [[0]], [1])
        onesBatchShape = tf.concat([onesBatchShape, [1,1]], 0)

        zeros = tf.zeros(zeroBatchShape, dtype=tf.complex128)
        ones = tf.ones(onesBatchShape, dtype=tf.complex128)

        if (sign == +1):
            boundaryMask = tf.concat([zeros, ones], 0)
        else:
            boundaryMask = tf.concat([ones, zeros], 0)

        return boundaryMask

    def gaugeBoundaryMask(self, sign):
        zeroBatchShape = self.latShape
        zeroBatchShape = tf.tensor_scatter_nd_update(zeroBatchShape, [[0]], [zeroBatchShape[0] - 1])
        zeroBatchShape = tf.concat([zeroBatchShape, [3,2,2]], 0)
        
        eyeBatchShape = self.latShape
        eyeBatchShape = tf.tensor_scatter_nd_update(eyeBatchShape, [[0]], [1])
        eyeBatchShape = tf.concat([eyeBatchShape, [3]], 0)

        zeros = tf.zeros(zeroBatchShape, dtype=tf.complex128)
        identities = tf.eye(2, batch_shape=eyeBatchShape, dtype=tf.complex128)

        if (sign == +1):
            boundaryMask = tf.concat([zeros, identities], 0)
        else:
            boundaryMask = tf.concat([identities, zeros], 0)

        return boundaryMask

    def rGaugeBoundaryMask(self):
        yzZeros = tf.zeros(tf.concat([self.latShape, [2, 2]], 0), dtype=tf.complex128)
        zeroBatchShape = self.latShape
        zeroBatchShape = tf.tensor_scatter_nd_update(zeroBatchShape, [[0]], [zeroBatchShape[0] - 1])
        zeroBatchShape = tf.concat([zeroBatchShape, [2,2]], 0)
        
        eyeBatchShape = self.latShape
        eyeBatchShape = tf.tensor_scatter_nd_update(eyeBatchShape, [[0]], [1])

        rZeros = tf.zeros(zeroBatchShape, dtype=tf.complex128)
        rIdentites = tf.eye(2, batch_shape=eyeBatchShape, dtype=tf.complex128)

        rMask = tf.concat([rIdentites, rZeros], 0)

        boundaryMask = tf.stack([rMask, yzZeros, yzZeros], -3)

        return boundaryMask

    def plaquetteBoundaryMask(self, sign):
        zeroBatchShape = self.latShape
        zeroBatchShape = tf.tensor_scatter_nd_update(zeroBatchShape, [[0]], [zeroBatchShape[0] - 1])
        zeroBatchShape = tf.concat([zeroBatchShape, [2,2]], 0)
        
        eyeBatchShape = self.latShape
        eyeBatchShape = tf.tensor_scatter_nd_update(eyeBatchShape, [[0]], [1])

        zeros = tf.zeros(zeroBatchShape, dtype=tf.complex128)
        identities = tf.eye(2, batch_shape=eyeBatchShape, dtype=tf.complex128)

        if (sign == +1):
            boundaryMask = tf.concat([zeros, identities], 0)
        else:
            boundaryMask = tf.concat([identities, zeros], 0)

        return boundaryMask

    # Process gradients so the respect the field constraints. Includes division by
    # 2*pi*R to make gradients comparable to those in 3d cartesian theories
    def processGradients(self, grads, fields):
        processedGrads = grads

        processedGrads[0] = grads[0] / (2*np.pi*tf.cast(
            tf.reshape(self.metric, tf.concat([self.latShape, [1, 1]], 0)), tf.complex128)
            )
        processedGrads[1] = FieldTools.projectSu2Gradients(grads[1], fields[1])
        processedGrads[1] = processedGrads[1] / (2*np.pi*tf.cast(
            tf.reshape(self.metric, tf.concat([self.latShape, [1, 1, 1]], 0)), tf.complex128)
            )

        return processedGrads