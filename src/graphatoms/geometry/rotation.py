from warnings import catch_warnings, filterwarnings

import numpy as np
from numpy import typing as npt
from scipy.spatial.transform import Rotation as Rot


def rotate(
    points: npt.ArrayLike,
    rotation: Rot = Rot.random(),
    center: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Rotate 3D points by the provided rotation around a given center.

    Default rotation center is the geometry center of the given points.

    Args:
        points (npt.ArrayLike): The 3D points to rotate.
        rotation (Rot, optional): The rotation. Defaults to Rot.random().
        center (npt.ArrayLike, optional): The center of the rotation.
            Defaults to None which means that rotation will take
                place around the geometry center of input points.

    Returns:
        np.ndarray: The points after rotation.
    """
    assert isinstance(rotation, Rot), (
        "Rotation must be an instance of scipy.spatial.transform.Rotation."
    )

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Points must be an array of 3D points.")

    if center is None:
        center = points.mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)
    assert (
        isinstance(center, np.ndarray)
        and center.ndim == 1
        and center.shape[0] == 3
    ), "Center must be a 3D vector."

    return rotation.apply(points - center) + center


def kabsch(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
) -> tuple[Rot, np.ndarray, float]:
    """Find rotation matrix and translation vector for convert A into B.

    This method solves and returns the R & t in
        A = rotate(B) + t
        where A & B are Nx3 matrices for point coordinates,
            R is the scipy Rotation object, and
            t is the 3D translation vector.

    See details:
        https://nghiaho.com/?page_id=671
        https://github.com/nghiaho12/rigid_transform_3D
        https://blog.csdn.net/u012836279/article/details/80203170
        https://blog.csdn.net/u012836279/article/details/80351462
    Advanced Methods:
        https://github.com/mammasmias/IterativeRotationsAssignments
        https://pubs.acs.org/doi/10.1021/acs.jcim.2c01187

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The first item is rotation matrix which is the instance
                instance of scipy.spatial.transform.Rotation.
            The second item is translation vector.
            The third item is the RMSD value.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Arrays must be of 2D arrays.")
    if A.shape[-1] != 3 or B.shape[-1] != 3:
        raise ValueError("Array elements must be 3D vectors.")
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)

    # find mean/centroid
    centroid_A: np.ndarray = np.mean(A, axis=0)
    centroid_B: np.ndarray = np.mean(B, axis=0)

    # subtract mean
    # NOTE: doing A -= centroid_A will modifiy input!
    A, B = A - centroid_A, B - centroid_B
    assert np.all(np.abs(np.mean(A, axis=0)) < 1e-5)
    assert np.all(np.abs(np.mean(B, axis=0)) < 1e-5)

    with catch_warnings():
        filterwarnings("ignore", category=UserWarning)
        rotation, rmsd = Rot.align_vectors(
            A,
            B,
            weights=None,
            return_sensitivity=False,
        )
    return rotation, centroid_A - centroid_B, rmsd


#######################################################################
#                                   Test
#######################################################################


def test_rotation_and_kabsch() -> None:  # noqa: D103
    import pytest
    from ase.build import molecule

    ben = molecule("C6H6")

    # -----------------------------------------------------------------
    # test for rotation
    points = ben.positions + np.random.random(3)
    center = points.mean(axis=0)
    print(f"Points center: {center}")

    rotation = Rot.random()
    print(f"Random rotation:\n{rotation.as_quat()}")

    rotated = rotate(points, rotation, center=[0, 0, 0])
    print(f"Rotated points around zero point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) != center

    rotated = rotate(points, rotation, center)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center

    # Default center point as the geometry center:
    rotated = rotate(points, rotation)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center

    # -----------------------------------------------------------------
    # test for  kabsch
    B = ben.positions + np.random.random(3)
    R, T = Rot.random(), np.random.random(3)

    A = rotate(B, R) + T
    assert pytest.approx((A - T).mean(axis=0)) == B.mean(axis=0)

    R0, T0, rmsd = kabsch(A, B)
    assert isinstance(T0, np.ndarray)
    assert pytest.approx(T0) == T
    assert isinstance(R0, Rot)
    assert pytest.approx(R0.as_matrix()) == R.as_matrix()
    assert rmsd <= 1e-5
    assert pytest.approx(rotate(B, R0) + T0) == A
