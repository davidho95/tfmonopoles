"""
Class for calculating field properties in Georgi-Glashow Su(2) Theory
"""

import tensorflow as tf
import numpy as np
from tfmonopoles import FieldTools

class GeorgiGlashowSu2Theory:
    # Params is a dictionary with keys "vev", "selfCoupling", "gaugeCoupling",
    # and optionally "boundaryConditions" and/or "tHooftLine".
    #
    # boundaryConditions is a list specifying the boundary conditions in each
    # coordinate direction: the number denotes the pauli matrix pre- and
    # postmultiplying the gauge field (0 is the identity).
    # If tHooftLine is specified, a line of plaquettes through the
    # centre of the lattice in the x direction is flipped.    
    def __init__(self, params):
        self.gaugeCoupling = params["gaugeCoupling"]
        self.vev = params["vev"]
        self.selfCoupling = params["selfCoupling"]
        
        if "boundaryConditions" in params:
            self.boundaryConditions = params["boundaryConditions"]
        else:
            self.boundaryConditions = [1, 2, 3] # Default to twisted BCs

        # Currently only 't Hooft lines in the x direction are supported
        if "tHooftLine" in params:
            self.tHooftLine = params["tHooftLine"]
        else:
            self.tHooftLine = False

    def energy(self, scalarField, gaugeField):
        return tf.math.reduce_sum(self.energyDensity(scalarField, gaugeField))

    def energyDensity(self, scalarField, gaugeField):
        energyDensity = tf.zeros(tf.shape(scalarField)[0:-2], dtype=tf.float64)

        energyDensity += self.ymTerm(gaugeField)
        energyDensity += self.covDerivTerm(scalarField, gaugeField)
        energyDensity += self.scalarPotential(scalarField)

        return energyDensity

    # Wilson action
    def ymTerm(self, gaugeField):
        energyDensity = tf.zeros(tf.shape(gaugeField)[0:3], dtype=tf.float64)

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 2/self.gaugeCoupling**2 * tf.math.real((2 - \
                    tf.linalg.trace(self.plaquette(gaugeField, ii, jj))))

        return energyDensity

    # Gauge kinetic term for the scalar field
    def covDerivTerm(self, scalarField, gaugeField):
        energyDensity = tf.zeros(tf.shape(scalarField)[0:-2], dtype=tf.float64)
        numDims = 3
        for ii in range(numDims):
            covDeriv = self.covDeriv(scalarField, gaugeField, ii)
            energyDensity += tf.math.real(tf.linalg.trace(covDeriv @ covDeriv))
        return energyDensity

    # Scalar potential
    def scalarPotential(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField)[0:-2], dtype=tf.float64)

        norms = tf.math.real(tf.linalg.trace(scalarField @ scalarField))
        energyDensity += self.selfCoupling * (self.vev**2 - norms)**2
        return energyDensity

    # Wilson plaquette on the lattice
    def plaquette(self, gaugeField, dir1, dir2):
        plaquette = gaugeField[:,:,:,dir1,:,:]
        plaquette = plaquette @\
            self.shiftGaugeField(gaugeField, dir1, +1)[:,:,:,dir2,:,:]
        plaquette = plaquette @ \
            tf.linalg.adjoint(
                self.shiftGaugeField(gaugeField, dir2, +1)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(gaugeField[:,:,:,dir2,:,:])


        # If 't Hooft line specified, flip a line of y-z plaquettes in the x 
        # direction
        if (dir1 != 0 and dir2 != 0 and self.tHooftLine):
            plaquette = self.flipPlaquette(plaquette)

        return plaquette

        return plaquette

    # Gauge covariant derivative
    def covDeriv(self, scalarField, gaugeField, dir):
        scalarFieldShifted = self.shiftScalarField(scalarField, dir, +1)
        covDeriv = gaugeField[:,:,:,dir,:,:] @ scalarFieldShifted @\
            tf.linalg.adjoint(gaugeField[:,:,:,dir,:,:]) - scalarField
        return covDeriv

    # Projects out abelian subgroup of gauge field
    def u1Projector(self, scalarField):
        trScalarSq = tf.linalg.trace(scalarField @ scalarField)
        latShape = tf.shape(trScalarSq)

        trScalarSq = tf.expand_dims(trScalarSq, -1)
        trScalarSq = tf.expand_dims(trScalarSq, -1)

        normalisedScalarField = tf.math.sqrt(2.0/trScalarSq) * scalarField

        identity = tf.eye(2, batch_shape=latShape, dtype=tf.complex128) 

        u1Projector = 0.5*(identity + normalisedScalarField)

        return u1Projector

    # Projects out abelian subgroup of gauge field
    def getU1Link(self, gaugeField, scalarField, dir):
        projector = self.u1Projector(scalarField)
        projectorShifted = self.u1Projector(
            self.shiftScalarField(scalarField, dir, +1)
            )

        u1Link = projector @ gaugeField[:,:,:,dir,:,:] @ projectorShifted

        return u1Link

    # Plaquette formed from abelian links
    def u1Plaquette(self, gaugeField, scalarField, dir1, dir2):
        u1Plaquette = self.getU1Link(gaugeField, scalarField, dir1)
        u1Plaquette = u1Plaquette @ self.getU1Link(
            self.shiftGaugeField(gaugeField, dir1, +1), \
                self.shiftScalarField(scalarField, dir1, +1), dir2
            )
        u1Plaquette = u1Plaquette @ tf.linalg.adjoint(
            self.getU1Link(
                self.shiftGaugeField(gaugeField, dir2, +1),\
                    self.shiftScalarField(scalarField, dir2, +1), dir1
                )
            )
        u1Plaquette = u1Plaquette @ tf.linalg.adjoint(
            self.getU1Link(gaugeField, scalarField, dir2)
            )

        return u1Plaquette

    def magneticField(self, gaugeField, scalarField, dir):
        dir1 = (dir + 1) % 3
        dir2 = (dir + 2) % 3

        magneticField = tf.math.angle(
            tf.linalg.trace(
                self.u1Plaquette(gaugeField, scalarField, dir1, dir2))
            )

        if (dir == 0 and self.tHooftLine):
            # Correct values along the 't Hooft line
            latShape = tf.shape(magneticField)
            indices = self.tHooftLineIndices(latShape)

            updates = tf.gather_nd(magneticField, indices)
            updates = updates - np.pi * tf.sign(updates)

            magneticField = tf.tensor_scatter_nd_update(
                magneticField, indices, updates
                )
        return 2.0/self.gaugeCoupling * magneticField

    # Shifts scalar field using supplied BC's
    def shiftScalarField(self, scalarField, dir, sign):
        scalarFieldShifted = tf.roll(scalarField, -sign, dir)

        pauliMatNum = self.boundaryConditions[dir]

        if pauliMatNum == 0:
            return scalarFieldShifted

        latShape = tf.shape(scalarField)[0:3]
        indices = FieldTools.boundaryIndices(latShape, dir, sign)

        updates = tf.gather_nd(scalarFieldShifted, indices)
        updates = -1.0*FieldTools.pauliMatrix(pauliMatNum) @ updates @ FieldTools.pauliMatrix(pauliMatNum)

        scalarFieldShifted = tf.tensor_scatter_nd_update(scalarFieldShifted, indices, updates)
        return scalarFieldShifted

    # Shifts gauge field using supplied BC's
    def shiftGaugeField(self, gaugeField, dir, sign):
        gaugeFieldShifted = tf.roll(gaugeField, -sign, dir)

        pauliMatNum = self.boundaryConditions[dir]

        if pauliMatNum == 0:
            return gaugeFieldShifted

        latShape = tf.shape(gaugeField)[0:3]
        indices = FieldTools.boundaryIndices(latShape, dir, sign)

        updates = tf.gather_nd(gaugeFieldShifted, indices)
        updates = FieldTools.pauliMatrix(pauliMatNum) @ updates @ FieldTools.pauliMatrix(pauliMatNum)

        gaugeFieldShifted = tf.tensor_scatter_nd_update(gaugeFieldShifted, indices, updates)
        return gaugeFieldShifted


    def flipPlaquette(self, plaquette):
        # Mask to flip plaquettes along a line in the x direction, giving a
        # 't Hooft line
        latShape = tf.shape(plaquette)[0:3]
        indices = self.tHooftLineIndices(latShape)

        updates = -1.0*tf.gather_nd(plaquette, indices)
        plaquette = tf.tensor_scatter_nd_update(plaquette, indices, updates)

        return plaquette

    # Mask of -1 values along a tHooft line with ones elsewhere to flip plaquettes
    def tHooftLineMask(self, plaquette):
        latShape = tf.shape(plaquette)[0:3]

        mask = -tf.ones([latShape[0], 1, 1, 1, 1], dtype=tf.complex128)

        yPaddings = [latShape[1] // 2, latShape[1] - latShape[1] // 2 - 1]
        zPaddings = [latShape[2] // 2, latShape[2] - latShape[2] // 2 - 1]

        paddings = [[0,0], yPaddings, zPaddings, [0,0], [0,0]]

        mask = tf.pad(mask, paddings, constant_values=1)

        return mask

    # Indices of the 't Hooft line
    def tHooftLineIndices(self, latShape):
        indices = tf.stack(tf.meshgrid(
            tf.range(latShape[0]), latShape[1] //2, latShape[1] // 2, indexing="ij"
            ), -1)

        return indices

    # Postprocess the gauge gradients so they obey the constraints on the fields
    # Expects grad to be the output of tf.GradientTape.gradient()
    # Expects fields to be a list [scalarField, gaugeField]
    def processGradients(self, grads, fields):
        processedGrads = grads
        processedGrads[1] = FieldTools.projectSu2Gradients(grads[1], fields[1])

        return processedGrads

