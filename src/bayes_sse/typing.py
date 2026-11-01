from typing import Literal

import numpy as np
from numpy.typing import NDArray

# General-purpose array types
type Float64Array = NDArray[np.float64]
type Float64FlatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
type Float64Vector = np.ndarray[tuple[int, Literal[1]], np.dtype[np.float64]]

type Float64Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Float64MatrixBatch = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

# Domain-specific

# Individual samples: (dimension, 1)

type StateVector = Float64Vector
type ObservationVector = Float64Vector
type NoiseVector = Float64Vector
type ControlVector = Float64Vector | None

# Batches: (dimension, n_samples)
type StateBatch = Float64Matrix
type ObservationBatch = Float64Matrix
type NoiseBatch = Float64Matrix

# Covariance
type CovarianceMatrix = Float64Matrix
type CovarianceFactor = Float64Matrix
type CrossCovarianceMatrix = Float64Matrix

# Particle representation
type ParticleMatrix = StateBatch
type ParticleWeights = Float64FlatVector
type ParticleIndices = np.ndarray[tuple[int], np.dtype[np.intp]]

# Numerical integration
type IntegrationPoints = Float64Matrix
type IntegrationWeights = Float64FlatVector

# Mixture components occupy the first axis
type MixtureMeans = Float64Matrix
type MixtureCovariances = Float64MatrixBatch
type MixtureCovarianceFactors = Float64MatrixBatch
type MixtureWeights = Float64FlatVector
