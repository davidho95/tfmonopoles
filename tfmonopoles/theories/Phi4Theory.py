"""
Class for calculating field properties in scalar Phi^4 theory
"""

import tensorflow as tf

class Phi4Theory:
    # Params is a dictionary with keys "vev", "selfCoupling"
    # indicating the parameters of the theory
    def __init__(self, params):
        self.vev = params["vev"]
        self.selfCoupling = params["selfCoupling"]

    def energy(self, scalarField):
        return tf.math.reduce_sum(self.energyDensity(scalarField))

    def energyDensity(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField), dtype=tf.float64)

        energyDensity += self.kineticTerm(scalarField)
        energyDensity += self.scalarPotential(scalarField)

        return energyDensity

    # Kinetic term for the scalar field
    def kineticTerm(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField), dtype=tf.float64)
        numDims = 3
        for ii in range(numDims):
            deriv = self.forwardDiff(scalarField, ii)
            energyDensity += tf.math.real(deriv * tf.math.conj(deriv))
        return energyDensity

    # Scalar potential
    def scalarPotential(self, scalarField):
        energyDensity = tf.zeros(tf.shape(scalarField), dtype=tf.float64)

        normSq = tf.math.real(scalarField * tf.math.conj(scalarField))
        energyDensity += self.selfCoupling * (self.vev**2 - normSq)**2
        return energyDensity

    # Finite-difference forward derivative
    def forwardDiff(self, scalarField, cpt):
        scalarFieldShifted = self.shiftScalarField(scalarField, cpt, +1)
        deriv = scalarFieldShifted - scalarField
        return deriv

    # Shifts scalar field using periodic BC's
    def shiftScalarField(self, scalarField, cpt, sign):
        scalarFieldShifted = tf.roll(scalarField, -sign, cpt)

        return scalarFieldShifted