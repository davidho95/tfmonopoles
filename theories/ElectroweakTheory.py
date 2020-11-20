'''
Class for calculating field properties in Electroweak theory
'''

import tensorflow as tf
import numpy as np
import FieldTools

class ElectroweakTheory:
    # Params is a dictionary with keys "vev", "selfCoupling", "gaugeCoupling", and "mixingAngle"
    def __init__(self, params):
        self.gaugeCoupling = tf.cast(params["gaugeCoupling"], dtype=tf.float64)
        self.vev = tf.cast(params["vev"], dtype=tf.float64)
        self.selfCoupling = tf.cast(params["selfCoupling"], dtype=tf.float64)
        self.tanSqMixingAngle = tf.cast(tf.math.tan(params["mixingAngle"])**2, dtype=tf.float64)

    def energy(self, higgsField, isospinField, hyperchargeField):
        return tf.math.reduce_sum(self.energyDensity(higgsField, isospinField, hyperchargeField))

    # Higgs field is an [N, N, N, 2, 1] tensor
    # Isospin field is a [N, N, N, 3, 2, 2] tensor
    # Hypercharge field is a [N, N, N, 3, 1, 1] tensor
    # Trailing dimensions are for broadcasting
    def energyDensity(self, higgsField, isospinField, hyperchargeField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)

        energyDensity += self.isospinYMTerm(isospinField)
        energyDensity += self.hyperchargeYMTerm(hyperchargeField)
        energyDensity += self.covDerivTerm(higgsField, isospinField, hyperchargeField)
        energyDensity += self.scalarPotential(higgsField)

        return energyDensity

    # Wilson action
    def isospinYMTerm(self, isospinField):
        energyDensity = tf.zeros(tf.shape(isospinField)[0:3], dtype=tf.float64)

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 2/self.gaugeCoupling**2 * tf.math.real((2 - \
                    tf.linalg.trace(self.isospinPlaquette(isospinField, ii, jj))))

        return energyDensity

    # Compact U(1) formalism
    def hyperchargeYMTerm(self, hyperchargeField):
        energyDensity = tf.zeros(tf.shape(hyperchargeField)[0:3], dtype=tf.float64)

        # print(tf.math.real((1 - \
        #             tf.linalg.trace(self.hyperchargePlaquette(hyperchargeField, 0, 1)))))

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 1/(self.gaugeCoupling**2*self.tanSqMixingAngle) * tf.math.real((1 - \
                    tf.linalg.trace(self.hyperchargePlaquette(hyperchargeField, ii, jj))))

        return energyDensity

    # Gauge kinetic term for the scalar field
    def covDerivTerm(self, higgsField, isospinField, hyperchargeField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)
        numDims = 3
        for ii in range(numDims):
            covDeriv = self.covDeriv(higgsField, isospinField, hyperchargeField, ii)
            energyDensity += tf.math.real(tf.linalg.trace(covDeriv @ tf.linalg.adjoint(covDeriv)))
        return energyDensity

    # Scalar potential
    def scalarPotential(self, higgsField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)

        magnitudeSq = tf.math.real(tf.linalg.trace(higgsField @ tf.linalg.adjoint(higgsField)))
        energyDensity += self.selfCoupling * (magnitudeSq - 0.5*self.vev**2)**2
        return energyDensity

    # SU(2) Wilson plaquette on the lattice
    def isospinPlaquette(self, isospinField, dir1, dir2):
        plaquette = isospinField[:,:,:,dir1,:,:]
        plaquette = plaquette @ \
            self.shiftIsospinField(isospinField, dir1, sign)[:,:,:,dir2,:,:]
        plaquette = plaquette @ \
            tf.linalg.adjoint(
                self.shiftIsospinField(isospinField, dir2, sign)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(isospinField[:,:,:,dir2,:,:])

        return plaquette

    # U(1) Wilson plaquette on the lattice
    def hyperchargePlaquette(self, hyperchargeField, dir1, dir2):
        plaquette = hyperchargeField[:,:,:,dir1,:,:]
        plaquette = plaquette @ \
            self.shiftHyperchargeField(hyperchargeField, dir1, sign)[:,:,:,dir2,:,:]
        plaquette = plaquette @ \
            tf.linalg.adjoint(
                self.shiftHyperchargeField(hyperchargeField,dir2, sign)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(hyperchargeField[:,:,:,dir2,:,:])

        return plaquette

    # Gauge covariant derivative
    def covDeriv(self, higgsField, isospinField, hyperchargeField, dir):
        higgsFieldShifted = self.shiftHiggsField(higgsField, dir, sign)
        covDeriv = hyperchargeField[:,:,:,dir,:,:] *\
            isospinField[:,:,:,dir,:,:] @\
            higgsFieldShifted - higgsField
        return covDeriv

    # Magnitude of Higgs field
    def higgsMagnitude(self, higgsField):
        higgsMagnitudeSq = tf.linalg.adjoint(higgsField) @ higgsField
        return tf.math.sqrt(higgsMagnitudeSq)

    # Projects out abelian (electromagnetic) subgroup of gauge fields
    def getEmLink(self, isospinField, hyperchargeField, higgsField, dir):
        higgsFieldShifted = self.shiftHiggsField(higgsField, dir, sign)
        emLink = higgsField @\
            tf.linalj.adjoint(hyperchargeField[:,:,:,dir,:,:]) @\
            isospinField[:,:,:,dir,:,:] @ higgsFieldShifted /\
            (higgsMagnitude(higgsField) * higgsMagnitude(higgsFieldShifted))

        return emLink

    # Plaquette formed from abelian links
    def emPlaquette(self, isospinField, hyperchargeField, higgsField, dir1, dir2):
        emPlaquette = self.getEmLink(isospinField, hyperchargeField, higgsField, dir1)
        emPlaquette = emPlaquette @ self.getEmLink(
            self.shiftIsospinField(isospinField, dir1, sign), \
                self.shiftHyperchargeField(hyperchargeField, dir1, sign), \
                self.shiftHiggsField(higgsField, dir1, sign), dir2
            )
        emPlaquette = emPlaquette @ tf.linalg.adjoint(
            self.getEmLink(
                self.shiftIsospinField(isospinField, dir2, sign), \
                self.shiftHyperchargeField(hyperchargeField, dir2, sign), \
                    self.shiftHiggsField(higgsField, dir2, sign), dir1
                )
            )
        emPlaquette = emPlaquette @ tf.linalg.adjoint(
            self.getEmLink(isospinField, higgsField, hyperchargeField, dir2)
            )

        return emPlaquette

    def magneticField(self, isospinField, higgsField, dir):
        dir1 = (dir + 1) % 3
        dir2 = (dir + 2) % 3

        magneticField = 2.0/(self.gaugeCoupling * sqrt(self.tanSqMixingAngle)) * \
            tf.math.angle(
                tf.linalg.trace(
                    self.emPlaquette(isospinField, hyperchargeField, higgsField, dir1, dir2)
                )
            )

        return magneticField

    # Shifts scalar field using periodic BCs
    def shiftHiggsField(self, higgsField, dir, sign):
        shiftedField = tf.roll(higgsField, sign, dir)
        return shiftedField

    # Shifts isospin field using periodic BCs
    def shiftIsospinField(self, isospinField, dir, sign):
        shiftedField = tf.roll(isospinField, sign, dir)
        return shiftedField

    # Shifts hypercharge field using periodic BCs
    def shiftHyperchargeField(self, hyperchargeField, dir, sign):
        shiftedField = tf.roll(hyperchargeField, sign, dir)
        return shiftedField