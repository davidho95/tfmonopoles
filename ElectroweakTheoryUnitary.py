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

    # Higgs field is an [N, N, N, 1, 1] complex128 tensor,
    #   but only the real part is used
    # Isospin field is a [N, N, N, 3, 2, 2] complex128 tensor
    # Hypercharge field is a [N, N, N, 3, 1, 1] complex128 tensor
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
        higgsMagnitude = tf.linalg.trace(self.higgsMagnitude(tf.math.real(higgsField)))
        numDims = 3
        for ii in range(numDims):
            higgsFieldShifted = self.shiftHiggsField(higgsField, ii)
            higgsMagnitudeShifted = tf.linalg.trace(self.higgsMagnitude(tf.math.real(higgsFieldShifted)))
            energyDensity += higgsMagnitude**2
            energyDensity += higgsMagnitudeShifted**2
            energyDensity -= higgsMagnitude * higgsMagnitudeShifted *\
                tf.math.real(tf.linalg.trace(isospinField[:,:,:,ii,:,:])) *\
                tf.math.real(tf.linalg.trace(hyperchargeField[:,:,:,ii,:,:]))
            energyDensity += higgsMagnitude * higgsMagnitudeShifted *\
                tf.math.imag(tf.linalg.trace(isospinField[:,:,:,ii,:,:] @ FieldTools.pauliMatrix(2))) *\
                tf.math.imag(tf.linalg.trace(hyperchargeField[:,:,:,ii,:,:]))


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
            self.shiftIsospinField(isospinField, dir1)[:,:,:,dir2,:,:]
        plaquette = plaquette @ \
            tf.linalg.adjoint(
                self.shiftIsospinField(isospinField,dir2)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(isospinField[:,:,:,dir2,:,:])

        return plaquette

    # U(1) Wilson plaquette on the lattice
    def hyperchargePlaquette(self, hyperchargeField, dir1, dir2):
        plaquette = hyperchargeField[:,:,:,dir1,:,:]
        plaquette = plaquette @ \
            self.shiftHyperchargeField(hyperchargeField, dir1)[:,:,:,dir2,:,:]
        plaquette = plaquette @ \
            tf.linalg.adjoint(
                self.shiftHyperchargeField(hyperchargeField,dir2)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(hyperchargeField[:,:,:,dir2,:,:])

        return plaquette

    # Gauge covariant derivative
    def covDeriv(self, higgsField, isospinField, hyperchargeField, dir):
        higgsFieldShifted = self.shiftHiggsField(higgsField, dir)
        zeroMat = tf.zeros(tf.shape(higgsField), dtype=tf.complex128)
        higgsFieldVec = tf.cast(tf.math.real(tf.concat([zeroMat, higgsField], -2)), tf.complex128)
        higgsFieldVecShifted = tf.cast(tf.math.real(tf.concat([zeroMat, higgsFieldShifted], -2)), tf.complex128)
        covDeriv = hyperchargeField[:,:,:,dir,:,:] *\
            isospinField[:,:,:,dir,:,:] @\
            higgsFieldVecShifted - higgsFieldVec
        return covDeriv

    # Magnitude of Higgs field
    def higgsMagnitude(self, higgsField):
        higgsMagnitudeSq = tf.linalg.adjoint(higgsField) @ higgsField
        return tf.math.sqrt(higgsMagnitudeSq)

    # Projects out abelian (electromagnetic) subgroup of gauge fields
    def getEmLink(self, isospinField, hyperchargeField, higgsField, dir):
        higgsFieldShifted = self.shiftHiggsField(higgsField, dir)
        emLink = higgsField @\
            tf.linalj.adjoint(hyperchargeField[:,:,:,dir,:,:]) @\
            isospinField[:,:,:,dir,:,:] @ higgsFieldShifted /\
            (higgsMagnitude(higgsField) * higgsMagnitude(higgsFieldShifted))

        return emLink

    # Plaquette formed from abelian links
    def emPlaquette(self, isospinField, hyperchargeField, higgsField, dir1, dir2):
        emPlaquette = self.getEmLink(isospinField, hyperchargeField, higgsField, dir1)
        emPlaquette = emPlaquette @ self.getEmLink(
            self.shiftIsospinField(isospinField, dir1), \
                self.shiftHyperchargeField(hyperchargeField, dir1), \
                self.shiftHiggsField(higgsField, dir1), dir2
            )
        emPlaquette = emPlaquette @ tf.linalg.adjoint(
            self.getEmLink(
                self.shiftIsospinField(isospinField, dir2), \
                self.shiftHyperchargeField(hyperchargeField, dir2), \
                    self.shiftHiggsField(higgsField, dir2), dir1
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
    def shiftHiggsField(self, higgsField, dir):
        shiftedField = tf.roll(higgsField, -1, dir)
        return shiftedField

    # Shifts isospin field using periodic BCs
    def shiftIsospinField(self, isospinField, dir):
        shiftedField = tf.roll(isospinField, -1, dir)
        return shiftedField

    # Shifts hypercharge field using periodic BCs
    def shiftHyperchargeField(self, hyperchargeField, dir):
        shiftedField = tf.roll(hyperchargeField, -1, dir)
        return shiftedField