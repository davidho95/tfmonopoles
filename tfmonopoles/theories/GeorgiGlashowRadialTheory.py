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
        scalarFieldShifted = tf.roll(scalarField, -sign, dir)

        if dir != 0:
            return scalarFieldShifted

        # Apply reflecting BC's by setting links at the boundary to
        # corresponding values from unshifted field
        indices = FieldTools.boundaryIndices(self.latShape, dir, sign)
        updates = tf.gather_nd(scalarField, indices)
        scalarFieldShifted = tf.tensor_scatter_nd_update(
            scalarFieldShifted, indices, updates
            )

        return scalarFieldShifted
    # Shifts gauge field using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards. As there is a physical boundary, shifting in one
    # direction then another is not an identity operation.
    def shiftGaugeField(self, gaugeField, dir, sign):
        # Moving one site forwards is equivalent to shifting the whole field
        # backwards, hence the minus sign (active/passive transform)
        gaugeFieldShifted = tf.roll(gaugeField, -sign, dir)

        if dir != 0:
            return gaugeFieldShifted

        # Apply reflecting BC's by setting links at the boundary to
        # corresponding values from unshifted field
        indices = FieldTools.boundaryIndices(self.latShape, dir, sign)
        updates = tf.gather_nd(gaugeField, indices)
        gaugeFieldShifted = tf.tensor_scatter_nd_update(
            gaugeFieldShifted, indices, updates
            )

        if sign == -1:
        # Set the r-links at the origin to the identity
            rOriginIndices = tf.stack(
                tf.meshgrid(0, tf.range(self.latShape[1]), tf.range(self.latShape[2]), 0, indexing="ij"), -1
                )
            rOriginUpdates = tf.eye(2, batch_shape=tf.shape(rOriginIndices)[0:-1], dtype=tf.complex128)
            gaugeFieldShifted = tf.tensor_scatter_nd_update(
                gaugeFieldShifted, rOriginIndices, rOriginUpdates
                )

        return gaugeFieldShifted

    # Shifts plaquette using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards 
    def shiftPlaquette(self, plaquette, plaqDir1, plaqDir2, shiftDir, sign):
        # Moving one site forwards is equivalent to shifting the whole field
        # backwards, hence the minus sign (active/passive transform)
        plaquetteShifted = tf.roll(plaquette, -sign, shiftDir)

        if shiftDir != 0:
            return plaquetteShifted

        indices = FieldTools.boundaryIndices(self.latShape, shiftDir, sign)

        if sign == -1 and (plaqDir1 == 0 or plaqDir2 == 0):
        # Set the plaquettes straddling the origin to the identity
            updates = tf.eye(2, batch_shape=tf.shape(indices)[0:-1], dtype=tf.complex128)
            plaquetteShifted = tf.tensor_scatter_nd_update(
                plaquetteShifted, indices, updates
                )
        else:
        # Apply reflecting BC's by setting links at the boundary to
        # corresponding values from unshifted field
            updates = tf.gather_nd(plaquette, indices)
            plaquetteShifted = tf.tensor_scatter_nd_update(
                plaquetteShifted, indices, updates
                )

        return plaquetteShifted

    # Shifts covariant derivative using periodic (y,z) or reflecting (r) BC's.
    # dir indicates the direction (r,y,z) of the shift and sign (+-1) indicates
    # forward or backwards 
    #
    # Assumes derivative direction is the same as dir
    def shiftCovDeriv(self, covDeriv, dir, sign):
        covDerivShifted = tf.roll(covDeriv, -sign, dir)

        if dir != 0:
            return covDerivShifted

        indices = FieldTools.boundaryIndices(self.latShape, dir, sign)

        if sign == -1:
            updates = tf.zeros(tf.concat([tf.shape(indices)[0:-1], [2,2]], 0), dtype=tf.complex128)
        else:
            updates = tf.gather_nd(covDeriv, indices)

        covDerivShifted = tf.tensor_scatter_nd_update(
            covDerivShifted, indices, updates
            )

        return covDerivShifted


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