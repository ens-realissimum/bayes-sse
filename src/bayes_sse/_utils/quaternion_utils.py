"""
Quaternion utilities using scipy.spatial.transform.Rotation.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class Quaternion:
    """
    Quaternion class.
    """

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initialize quaternion with w, x, y, z components.

        :param w: scalar part
        :param x: x component
        :param y: y component
        :param z: z component
        """
        self._wxyz = np.array([float(w), float(x), float(y), float(z)], dtype=np.float64)
        self._rotation = None

    @property
    def w(self) -> float:
        """Get scalar part of quaternion."""
        return float(self._wxyz[0])

    @property
    def x(self) -> float:
        """Get x component of quaternion."""
        return float(self._wxyz[1])

    @property
    def y(self) -> float:
        """Get y component of quaternion."""
        return float(self._wxyz[2])

    @property
    def z(self) -> float:
        """Get z component of quaternion."""
        return float(self._wxyz[3])

    @property
    def components(self) -> np.ndarray:
        """Get quaternion components as [w, x, y, z] array."""
        return self._wxyz.copy()

    @property
    def normalized(self) -> 'Quaternion':
        """Return normalized quaternion."""
        norm = np.linalg.norm(self._wxyz)
        if norm > 0:
            normalized = self._wxyz / norm
            result = Quaternion(normalized[0], normalized[1], normalized[2], normalized[3])
            return result

        return Quaternion(self._wxyz[0], self._wxyz[1], self._wxyz[2], self._wxyz[3])

    def get_rotation(self) -> Rotation:
        """Gets the rotation."""
        if self._rotation is None:
            self._rotation = Rotation.from_quat(np.array([self._wxyz[1], self._wxyz[2], self._wxyz[3], self._wxyz[0]]))

        return self._rotation

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Multiply two quaternions using Hamilton product.
        """
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply Quaternion with Quaternion")

        w1, x1, y1, z1 = self._wxyz
        w2, x2, y2, z2 = other._wxyz

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w, x, y, z)

    def __rmul__(self, scalar: float) -> 'Quaternion':
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Quaternion by scalar")

        scaled = self._wxyz * float(scalar)
        return Quaternion(scaled[0], scaled[1], scaled[2], scaled[3])

    def __repr__(self) -> str:
        return f"quaternion({self._wxyz[0]}, {self._wxyz[1]}, {self._wxyz[2]}, {self._wxyz[3]})"

    def __deepcopy__(self, memo):
        """Support deepcopy."""
        result = Quaternion(self._wxyz[0], self._wxyz[1], self._wxyz[2], self._wxyz[3])
        memo[id(self)] = result
        return result


def quaternion(w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Quaternion:
    """Create a quaternion with given components."""
    return Quaternion(w, x, y, z)


def from_float_array(arr: np.ndarray | list[int] | list[float]) -> Quaternion:
    """Create quaternion from float array [w, x, y, z]."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError("Array must have shape (4,) for [w, x, y, z]")
    return Quaternion(arr[0], arr[1], arr[2], arr[3])


def as_float_array(q: Quaternion) -> np.ndarray:
    """Convert quaternion to float array [w, x, y, z]."""
    return q.components


def rotate_vectors(q: Quaternion, vectors: np.ndarray) -> np.ndarray:
    """
    Rotate vectors using quaternion.
    :param q: quaternion representing rotation
    :param vectors: array of vectors to rotate (can be single vector or multiple vectors)
    :return: rotated vectors
    """
    q_norm = q.normalized
    rotation = q_norm.get_rotation()

    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim == 1:
        return rotation.apply(vectors)
    else:
        return rotation.apply(vectors)
