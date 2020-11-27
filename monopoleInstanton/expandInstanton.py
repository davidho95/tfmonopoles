"""
Takes an instanton configuration as input and returns the same solution on a
field twice the size in every direction. There is no change to the lattice
spacing; the lattice is simply extended in every direction. The number of flux
quanta in the input solution must be specified as an argument.
"""

import tensorflow as tf
from tfmonopoles import FieldTools
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Expand an instanton solution")
parser.add_argument("--outputPath", "-o", default="", type=str)
parser.add_argument("--inputPath", "-i", default="", type=str)
parser.add_argument("--fluxQuanta", "-B", default=0, type=int)

args = parser.parse_args()

# Load data from input path
inputPath = args.inputPath
inputR = tf.constant(np.load(inputPath + "/R.npy", allow_pickle=True))
inputY = tf.constant(np.load(inputPath + "/Y.npy", allow_pickle=True))
inputZ = tf.constant(np.load(inputPath + "/Z.npy", allow_pickle=True))
inputScalarField = np.load(inputPath + "/scalarField.npy", allow_pickle=True)
inputGaugeField = np.load(inputPath + "/gaugeField.npy", allow_pickle=True)
inputParams = np.load(inputPath + "/params.npy", allow_pickle=True).item()
inputLatShape = inputParams["latShape"]

yzPaddings = [[0,0], [1,1], [1,1], [0,0], [0,0]]

outputScalarField = inputScalarField
outputGaugeField = inputGaugeField

B = 10
smallMagneticField = FieldTools.constantMagneticField(
    inputR, inputY, inputZ, 0, -B
    )
# Subtract the original field so the padding works; this will be added back
outputGaugeField = FieldTools.linearSuperpose(
    outputGaugeField, smallMagneticField
    )

for ii in range(inputLatShape[1] // 2):
    outputScalarField = tf.pad(outputScalarField, yzPaddings, "symmetric")
    outputGaugeFieldR = tf.pad(
        outputGaugeField[:,:,:,0,:,:], yzPaddings, "symmetric"
        )
    outputGaugeFieldY = tf.pad(
        outputGaugeField[:,:,:,1,:,:], yzPaddings, "symmetric"
        )
    outputGaugeFieldZ = tf.pad(
        outputGaugeField[:,:,:,2,:,:], yzPaddings, "symmetric"
        )
    outputGaugeField = tf.stack(
        [outputGaugeFieldR, outputGaugeFieldY, outputGaugeFieldZ], -3
        )

rPaddings = [[0,1], [0,0], [0,0], [0,0], [0,0]]

for ii in range(inputLatShape[0]):
    outputScalarField = tf.pad(outputScalarField, rPaddings, "symmetric")
    outputGaugeFieldR = tf.pad(outputGaugeField[:,:,:,0,:,:], rPaddings, "symmetric")
    outputGaugeFieldY = tf.pad(outputGaugeField[:,:,:,1,:,:], rPaddings, "symmetric")
    outputGaugeFieldZ = tf.pad(outputGaugeField[:,:,:,2,:,:], rPaddings, "symmetric")
    outputGaugeField = tf.stack([outputGaugeFieldR, outputGaugeFieldY, outputGaugeFieldZ], -3)

outputLatShape = inputLatShape + inputLatShape

# Set up the lattice
r = tf.cast(
    tf.linspace(
        -1/2, tf.cast(outputLatShape[0], tf.float32) - 1/2, outputLatShape[0]
        ), tf.float64
    )
y = tf.cast(
    tf.linspace(
        -(outputLatShape[1]-1)/2, (outputLatShape[1]-1)/2, outputLatShape[1]
        ), tf.float64
    )
z = tf.cast(
    tf.linspace(
    -(outputLatShape[2]-1)/2, (outputLatShape[2]-1)/2, outputLatShape[2]
        ), tf.float64
    )

R,Y,Z = tf.meshgrid(r, y, z, indexing="ij")

print(tf.shape(R))
print(tf.shape(outputGaugeField))

outputMagneticField = FieldTools.constantMagneticField(R, Y, Z, 0, 4*B)
outputGaugeField = FieldTools.linearSuperpose(
    outputGaugeField, outputMagneticField
    )

outputParams = inputParams
outputParams["latShape"] = outputLatShape

print("Instanton expanded. Orginal size:")
print(tf.shape(inputR).numpy())
print("New size:")
print(tf.shape(R).numpy())

# Save field and output parameters
outputPath = args.outputPath
if outputPath != "":
    np.save(outputPath + "/R", R.numpy())
    np.save(outputPath + "/Y", Y.numpy())
    np.save(outputPath + "/Z", Z.numpy())
    np.save(outputPath + "/scalarField", outputScalarField.numpy())
    np.save(outputPath + "/gaugeField", outputGaugeField.numpy())
    np.save(outputPath + "/params", outputParams)