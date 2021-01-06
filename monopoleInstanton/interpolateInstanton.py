"""
Takes an instanton configuration as input and returns the same solution on a
field twice the size in every direction, by interpolating between each pair 
of points
"""

import tensorflow as tf
from tfmonopoles import FieldTools
from tfmonopoles.theories import GeorgiGlashowRadialTheory
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="Expand an instanton solution")
parser.add_argument("--outputPath", "-o", default="", type=str)
parser.add_argument("--inputPath", "-i", default="", type=str)
parser.add_argument("--fluxQuanta", "-B", default=0, type=int)

args = parser.parse_args()

N = 64

# Load data from input path
inputPath = "../output/instantonDataB9V0_6171"
inputR = tf.constant(np.load(inputPath + "/R.npy", allow_pickle=True))
inputY = tf.constant(np.load(inputPath + "/Y.npy", allow_pickle=True))
inputZ = tf.constant(np.load(inputPath + "/Z.npy", allow_pickle=True))
inputScalarField = np.load(inputPath + "/scalarField.npy", allow_pickle=True)
inputGaugeField = np.load(inputPath + "/gaugeField.npy", allow_pickle=True)
inputParams = np.load(inputPath + "/params.npy", allow_pickle=True).item()
theory = GeorgiGlashowRadialTheory(inputParams)

def interp1d(scalarField, gaugeField, axis, theory):
    inputLatShape = tf.shape(scalarField)[0:3]
    outputLatShape = inputLatShape
    outputLatShape = tf.tensor_scatter_nd_update(outputLatShape, [[axis]], [2*outputLatShape[axis]])

    outputScalarField = tf.zeros(tf.concat([outputLatShape, [1, 1]], 0), dtype=tf.complex128)
    outputGaugeField = tf.zeros(tf.concat([outputLatShape, [3, 2, 2]], 0), dtype=tf.complex128)

    inputIndexVectors = [tf.range(inputLatShape[0]), tf.range(inputLatShape[1]), tf.range(inputLatShape[2])]
    inputIndices = tf.stack(tf.meshgrid(inputIndexVectors[0], inputIndexVectors[1], inputIndexVectors[2], indexing="ij"), -1)

    outputIndexVectorsOdd = inputIndexVectors.copy()
    outputIndexVectorsOdd[axis] = 2*inputIndexVectors[axis] + 1
    outputIndicesOdd = tf.stack(tf.meshgrid(outputIndexVectorsOdd[0], outputIndexVectorsOdd[1], outputIndexVectorsOdd[2], indexing="ij"), -1)
    originalScalarVals = tf.gather_nd(scalarField, inputIndices)
    originalGaugeVals = tf.gather_nd(gaugeField, inputIndices)
    outputScalarField = tf.tensor_scatter_nd_update(outputScalarField, outputIndicesOdd, originalScalarVals)
    outputGaugeField = tf.tensor_scatter_nd_update(outputGaugeField, outputIndicesOdd, originalGaugeVals)

    scalarFieldShifted = theory.shiftScalarField(scalarField, axis, -1)
    gaugeFieldShifted = theory.shiftGaugeField(gaugeField, axis, -1)
    avgScalarField = 0.5*(scalarField + scalarFieldShifted)
    avgGaugeField = FieldTools.linearAverage(gaugeField, gaugeFieldShifted)

    interpScalarVals = tf.gather_nd(avgScalarField, inputIndices)
    interpGaugeVals = tf.gather_nd(avgGaugeField, inputIndices)
    outputIndexVectorsEven = inputIndexVectors.copy()
    outputIndexVectorsEven[axis] = 2*inputIndexVectors[axis]
    outputIndicesEven = tf.stack(tf.meshgrid(outputIndexVectorsEven[0], outputIndexVectorsEven[1], outputIndexVectorsEven[2], indexing="ij"), -1)
    outputScalarField = tf.tensor_scatter_nd_update(outputScalarField, outputIndicesEven, interpScalarVals)
    outputGaugeField = tf.tensor_scatter_nd_update(outputGaugeField, outputIndicesEven, interpGaugeVals)

    return outputScalarField, outputGaugeField

B = 9
smallMagneticField = FieldTools.constantMagneticField(
    inputR, inputY, inputZ, 0, -B
    )
inputGaugeField = FieldTools.linearSuperpose(
    inputGaugeField, smallMagneticField
    )

outputScalarField, outputGaugeField = interp1d(inputScalarField, inputGaugeField, 0, theory)
outputScalarField, outputGaugeField = interp1d(outputScalarField, outputGaugeField, 1, theory)
outputScalarField, outputGaugeField = interp1d(outputScalarField, outputGaugeField, 2, theory)

outputLatShape = tf.shape(outputScalarField)[0:3]
outputParams = inputParams
outputParams["vev"] /= 2
outputParams["latShape"] = outputLatShape

r = tf.cast(
    tf.linspace(
        1/2, tf.cast(outputLatShape[0], tf.float32) - 1/2, outputLatShape[0]
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

bigMagneticField = FieldTools.constantMagneticField(
    R, Y, Z, 0, B
    )
outputGaugeField = FieldTools.linearSuperpose(
    outputGaugeField, bigMagneticField
    )

outputScalarField = outputScalarField / 2

# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection="3d")
# ax1.plot_surface(R[:,:,0], Y[:,:,0], outputScalarField[:,:,31,0,0])
# plt.show()

theory = GeorgiGlashowRadialTheory(outputParams)

print(theory.energy(outputScalarField, outputGaugeField))
eDensity = theory.energyDensity(outputScalarField, outputGaugeField)
eDensity0 = eDensity / theory.metric
magR = theory.magneticField(outputGaugeField, outputScalarField, 0)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
ax1.plot_surface(R[:,:,0], Y[:,:,0], magR[:,:,16])
plt.show()

# # Save field and output parameters
# outputPath = "../output/instantonDataB40V0_2641"
# if outputPath != "":
#     np.save(outputPath + "/R", R.numpy())
#     np.save(outputPath + "/Y", Y.numpy())
#     np.save(outputPath + "/Z", Z.numpy())
#     np.save(outputPath + "/scalarField", outputScalarField.numpy())
#     np.save(outputPath + "/gaugeField", outputGaugeField.numpy())
#     np.save(outputPath + "/params", outputParams)