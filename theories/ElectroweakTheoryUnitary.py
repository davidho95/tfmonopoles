"""
Class for calculating field properties in Electroweak theory
in the unitary gauge
"""

import tensorflow as tf
import numpy as np
import FieldTools

class ElectroweakTheoryUnitary:
    # Params is a dictionary with keys "vev", "selfCoupling", "gaugeCoupling",
    # and "mixingAngle"
    def __init__(self, params):
        self.gaugeCoupling = tf.cast(params["gaugeCoupling"], dtype=tf.float64)
        self.vev = tf.cast(params["vev"], dtype=tf.float64)
        self.selfCoupling = tf.cast(params["selfCoupling"], dtype=tf.float64)
        self.tanSqMixingAngle = tf.cast(
            tf.math.tan(params["mixingAngle"])**2, dtype=tf.float64
            )

    def energy(self, higgsField, isospinField, hyperchargeField):
        return tf.math.reduce_sum(self.energyDensity(
            higgsField, isospinField, hyperchargeField)
        )

    # Higgs field is an [N, N, N, 1, 1] complex128 tensor,
    #   but only the real part is used
    # Isospin field is a [N, N, N, 3, 2, 2] complex128 tensor
    # Hypercharge field is a [N, N, N, 3, 1, 1] complex128 tensor
    # Trailing dimensions are for broadcasting
    def energyDensity(self, higgsField, isospinField, hyperchargeField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)

        energyDensity += self.isospinYMTerm(isospinField)
        energyDensity += self.hyperchargeYMTerm(hyperchargeField)
        energyDensity += self.covDerivTerm(
            higgsField, isospinField, hyperchargeField
            )
        energyDensity += self.scalarPotential(higgsField)

        return energyDensity

    # Wilson action
    def isospinYMTerm(self, isospinField):
        energyDensity = tf.zeros(tf.shape(isospinField)[0:3], dtype=tf.float64)

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 2/self.gaugeCoupling**2 * tf.math.real(
                    2 - tf.linalg.trace(
                        self.isospinPlaquette(isospinField, ii, jj))
                        )

        return energyDensity

    # Compact U(1) formalism
    def hyperchargeYMTerm(self, hyperchargeField):
        energyDensity = tf.zeros(
            tf.shape(hyperchargeField)[0:3], dtype=tf.float64
            )

        numDims = 3
        for ii in range(numDims):
            for jj in range(numDims):
                if ii >= jj: continue
                energyDensity += 1/(
                    self.gaugeCoupling**2*self.tanSqMixingAngle) *\
                        tf.math.real(
                        1 - tf.linalg.trace(
                            self.hyperchargePlaquette(hyperchargeField, ii, jj)
                            )
                        )

        return energyDensity

    # Gauge kinetic term for the scalar field
    def covDerivTerm(self, higgsField, isospinField, hyperchargeField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)
        higgsMagnitude = tf.linalg.trace(
            self.higgsMagnitude(tf.math.real(higgsField))
            )
        numDims = 3

        # This is only valid in the unitary gauge, but it keeps the isospin
        # gradients symmetric which is good for convergence to the saddle
        for ii in range(numDims):
            higgsFieldShifted = self.shiftHiggsField(higgsField, ii, +1)
            higgsMagnitudeShifted = tf.linalg.trace(self.higgsMagnitude(
                tf.math.real(higgsFieldShifted))
                )
            energyDensity += higgsMagnitude**2
            energyDensity += higgsMagnitudeShifted**2
            energyDensity -= higgsMagnitude * higgsMagnitudeShifted *\
                tf.math.real(tf.linalg.trace(isospinField[:,:,:,ii,:,:])) *\
                tf.math.real(tf.linalg.trace(hyperchargeField[:,:,:,ii,:,:]))
            energyDensity += higgsMagnitude * higgsMagnitudeShifted *\
                tf.math.imag(tf.linalg.trace(isospinField[:,:,:,ii,:,:] @\
                FieldTools.pauliMatrix(3))) *\
                tf.math.imag(tf.linalg.trace(hyperchargeField[:,:,:,ii,:,:]))


        return energyDensity

    # Scalar potential
    def scalarPotential(self, higgsField):
        energyDensity = tf.zeros(tf.shape(higgsField)[0:3], dtype=tf.float64)

        magnitudeSq = tf.math.real(
            tf.linalg.trace(higgsField @ tf.linalg.adjoint(higgsField))
            )
        energyDensity += self.selfCoupling * (magnitudeSq - 0.5*self.vev**2)**2
        return energyDensity

    # SU(2) Wilson plaquette on the lattice
    def isospinPlaquette(self, isospinField, dir1, dir2):
        plaquette = isospinField[:,:,:,dir1,:,:]
        plaquette = plaquette @\
            self.shiftIsospinField(isospinField, dir1, +1)[:,:,:,dir2,:,:]
        plaquette = plaquette @\
            tf.linalg.adjoint(
                self.shiftIsospinField(isospinField,dir2, +1)[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @ tf.linalg.adjoint(isospinField[:,:,:,dir2,:,:])

        return plaquette

    # U(1) Wilson plaquette on the lattice
    def hyperchargePlaquette(self, hyperchargeField, dir1, dir2):
        plaquette = hyperchargeField[:,:,:,dir1,:,:]
        plaquette = plaquette @\
            self.shiftHyperchargeField(hyperchargeField, dir1, +1)[:,:,:,dir2,:,:]
        plaquette = plaquette @\
            tf.linalg.adjoint(
                self.shiftHyperchargeField(
                    hyperchargeField, dir2, +1
                    )[:,:,:,dir1,:,:]
                )
        plaquette = plaquette @\
            tf.linalg.adjoint(hyperchargeField[:,:,:,dir2,:,:])

        return plaquette

    # Magnitude of Higgs field
    def higgsMagnitude(self, higgsField):
        higgsMagnitudeSq = tf.linalg.adjoint(higgsField) @ higgsField
        return tf.math.sqrt(higgsMagnitudeSq)

    # Projects out abelian (electromagnetic) subgroup of gauge fields
    def getEmLink(self, isospinField, hyperchargeField, dir):
        emLink = (tf.linalg.adjoint(hyperchargeField[:,:,:,dir,:,:]) *\
            isospinField[:,:,:,dir,:,:])[:,:,:,1,1]

        return emLink

    # Plaquette formed from abelian links
    def emPlaquette(self, isospinField, hyperchargeField, dir1, dir2):
        emPlaquette = self.getEmLink(isospinField, hyperchargeField, dir1)
        emPlaquette = emPlaquette * self.getEmLink(
            self.shiftIsospinField(isospinField, dir1, +1), \
                self.shiftHyperchargeField(hyperchargeField, dir1), \
                dir2
            )
        emPlaquette = emPlaquette * tf.math.conj(
            self.getEmLink(
                self.shiftIsospinField(isospinField, dir2, +1), \
                self.shiftHyperchargeField(hyperchargeField, dir2), \
                    dir1
                )
            )
        emPlaquette = emPlaquette * tf.math.conj(
            self.getEmLink(isospinField, hyperchargeField, dir2)
            )

        return emPlaquette

    def magneticField(self, isospinField, hyperchargeField, dir):
        dir1 = (dir + 1) % 3
        dir2 = (dir + 2) % 3

        magneticField = 2.0/(self.gaugeCoupling * tf.math.sqrt(
            self.tanSqMixingAngle)
            )*\
            tf.math.angle(
                self.emPlaquette(
                    isospinField, hyperchargeField, dir1, dir2
                )
            )

        return magneticField

    # Shifts scalar field using periodic BCs
    def shiftHiggsField(self, higgsField, dir, sign):
        shiftedField = tf.roll(higgsField, -sign, dir)
        return shiftedField

    # Shifts isospin field using periodic BCs
    def shiftIsospinField(self, isospinField, dir, sign):
        shiftedField = tf.roll(isospinField, -sign, dir)
        return shiftedField

    # Shifts hypercharge field using periodic BCs
    def shiftHyperchargeField(self, hyperchargeField, dir, sign):
        shiftedField = tf.roll(hyperchargeField, -sign, dir)
        return shiftedField