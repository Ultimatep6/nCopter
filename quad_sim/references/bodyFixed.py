from __future__ import annotations

import numpy as np

from quad_sim.orientation.quaternion import Quaternion
from quad_sim.funcs import _get_inertial_to_body


class BodyFixed:
    """
    The BodyFixed frame is RIGIDLY attached to the drone, so the vector in the BodyFrame never changes because it rotates with the BF basis vectors.
    """

    def __init__(
        self,
        X: float | int | np.integer | np.floating,
        Y: float | int | np.integer | np.floating,
        Z: float | int | np.integer | np.floating,
        flag: str = "position",
    ):
        # We initiate the frame of reference with orthogonal basis vectors
        # X -- Roll Axis pointing forward (North)
        # Y -- Pitch Axis pointing left of X (East)
        # Z -- Yaw Axis pointing down

        self._basisX = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self._basisY = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        self._basisZ = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        self.__flag_list = [
            "position",
            "velocity",
            "acceleration",
            "ang_velocity",
            "ang_acceleration",
            "force",
            "moment",
        ]

        self.flag = flag

        self.vec = [X, Y, Z]

    @classmethod
    def from_EarthFixed(cls, objB: object, quaternion: Quaternion, CM: object = None):
        """
        A constructor that converts an EarthFixed vector into a BodyFixed vector
        """

        if objB.__class__.__name__ != "EarthFixed":
            raise TypeError("obj_B must be an EarthFixed object")

        rotation_matrix = _get_inertial_to_body(quaternion.as_np())

        if objB.flag == "position":
            if CM.__class__.__name__ != "EarthFixed":
                raise TypeError("CM must be an EarthFixed object")

            vec_BF = rotation_matrix @ (
                objB.vec - CM.vec
            )  # pyright: ignore[reportAttributeAccessIssue]
        else:
            vec_BF = rotation_matrix @ objB.vec

        return cls.from_Array(vec_BF, objB.flag)

    @classmethod
    def from_Array(cls, arr: np.ndarray, flag: str = "position"):
        flag_list = [
            "position",
            "velocity",
            "acceleration",
            "ang_velocity",
            "ang_acceleration",
            "force",
            "moment",
        ]

        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a np.ndarray")

        if not isinstance(flag, str):
            raise TypeError("flag must be a string")

        if flag not in flag_list:
            raise ValueError(f"flag must be one of {' or '.join(flag_list)}")

        return cls(*arr.ravel(), flag=flag)

    def changeFlag(self, value: str):
        self.flag = value
        return self

    def __add__(self, other: BodyFixed | np.ndarray) -> BodyFixed:
        if isinstance(other, np.ndarray):
            if other.shape != self.vec.shape:
                raise TypeError(f"ndarray must have shape {self.vec.shape}")
            return BodyFixed.from_Array(self.vec + other, flag=self.flag)

        if isinstance(other, BodyFixed):
            # if self.flag != other.flag:
            # raise TypeError("Cannot subtract vectors with different flags")
            return BodyFixed.from_Array(self.vec + other.vec, flag=self.flag)

        raise TypeError("Operand must be BodyFixed or ndarray")

    def __sub__(self, other: BodyFixed | np.ndarray) -> BodyFixed:
        if isinstance(other, np.ndarray):
            if other.shape != self.vec.shape:
                raise TypeError(f"ndarray must have shape {self.vec.shape}")
            return BodyFixed.from_Array(self.vec - other, flag=self.flag)

        if isinstance(other, BodyFixed):
            # if self.flag != other.flag:
            # raise TypeError("Cannot subtract vectors with different flags")
            return BodyFixed.from_Array(self.vec - other.vec, flag=self.flag)

        raise TypeError("Operand must be BodyFixed or ndarray")

    def __mul__(self, other: BodyFixed | int | float | np.ndarray) -> BodyFixed:
        if isinstance(other, BodyFixed):
            return BodyFixed.from_Array(
                np.cross(self.vec, other.vec, axis=0), flag=self.flag
            )
        if isinstance(other, (int, float)):
            return BodyFixed.from_Array(self.vec * other, flag=self.flag)

        if isinstance(other, np.ndarray):
            return BodyFixed.from_Array(
                np.cross(self.vec, other, axis=0), flag=self.flag
            )

        raise TypeError("Operand must be BodyFixed, int, or float")

    def __eq__(self, other) -> bool:
        if not isinstance(other, BodyFixed):
            return NotImplemented

        return self.flag == other.flag and np.allclose(self.vec, other.vec, atol=1e-12)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__

    @property
    def T(self):
        return BodyFixed.from_Array(self.vec.T, flag=self.flag)

    @property
    def vec(self):
        return self._vec

    @vec.setter
    def vec(self, value):
        # Expect a tuple/list of three scalars
        if not (isinstance(value, (tuple, list)) and len(value) == 3):
            raise TypeError(
                "vec must be a tuple/list of three numeric values (x, y, z)"
            )

        x, y, z = value

        # Validate scalar types
        for v, name in zip((x, y, z), ("x", "y", "z")):
            if not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be a numeric scalar")

        # Build the canonical (3,1) float32 array
        arr = np.array([[x], [y], [z]], dtype=np.float32)

        self._vec = arr

    @property
    def flag(self):
        return self._flag

    @flag.setter
    def flag(self, value: str):
        if not isinstance(value, str):
            raise TypeError("flag must be a str")
        elif value not in self.__flag_list:
            raise TypeError(f"flag must be one of {' '.join(self.__flag_list)}")
        else:
            self._flag = value

    @property
    def _flag_list(self):
        return self.__flag_list

    @property
    def x(self):
        return self.vec[0]

    @property
    def y(self):
        return self.vec[1]

    @property
    def z(self):
        return self.vec[2]
