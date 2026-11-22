"""Fit a common grasp centre and two opposed wrists to each filmed handover.

Pixels are marked on the 640 x 360 recording at the source contact time.
Each row records the token and the centres of the two finger mounting plates.
The tracking camera is calibrated from the four corners of the board.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# piece: token xy, left and right plate xy, left jaw roll from horizontal (degrees)
MARKS = {
    10: ((309, 135), (281, 138), (338, 134), -7),
    0: ((329, 131), (298, 137), (365, 124), -9),
    26: ((318, 137), (275, 137), (360, 136), 87),
    20: ((332, 132), (293, 138), (369, 129), -8),
    12: ((344, 123), (314, 130), (372, 119), -8),
    21: ((322, 132), (279, 131), (379, 114), 83),
    4: ((289, 143), (255, 137), (321, 150), 84),
    23: ((290, 124), (249, 116), (314, 136), 78),
}


# Visible centres of the camera-facing elbow caps, measured at 960 x 540.
ELBOW_CAPS = {
    10: ((256, 222), (689, 198)), 0: ((266, 251), (781, 180)),
    26: ((144, 217), (805, 224)), 20: ((219, 226), (816, 180)),
    12: ((360, 217), (778, 136)), 21: ((206, 232), (804, 184)),
    4: ((147, 164), (609, 261)), 23: ((147, 155), (608, 249)),
}


def fit_exchange(piece, when, camera, arms, seed, reference=False):
    token, left, right, roll = MARKS[piece]
    i = np.abs(camera["time"] - when).argmin()
    basis = Rotation.from_rotvec(camera["rot"][i]).as_matrix() @ np.diag([1, -1, -1])
    intrinsics = camera["K"] * (2 / 3)

    def project(p):
        view = (p - camera["eyes"][i]) @ basis
        return view[..., :2] / view[..., 2, None] * intrinsics[0, 0] + intrinsics[:2, 2]

    def residual(x):
        centre = x[:3]
        axis = np.r_[1.0, x[3:5]]
        axis /= np.linalg.norm(axis)
        plates = centre + np.array([[-0.073], [0.073]]) * axis
        errors = [
            2 * (project(centre) - token),
            (project(plates) - [left, right]).ravel(),
            (centre - seed) * [54, 36, 44],
            x[3:5] * 14,
        ]
        for k, arm in enumerate(arms):
            tip, wrist = arm.fk(x[5 + 6 * k : 11 + 6 * k])
            if reference:
                visible = arm.elbow(camera["eyes"][i])
                errors.append(1.15 * (project(visible) - np.array(ELBOW_CAPS[piece][k]) * (2/3)))
            errors.extend(
                [10000 * (tip - centre), 310 * (wrist[:, 2] - axis * (1 - 2 * k))]
            )
        return np.concatenate(errors)

    seed = np.clip(seed, [-0.052, -0.185, 0.222], [0.052, 0.02, 0.314])
    initial = np.r_[
        seed,
        [0, 0],
        *[arm.solve(seed, np.array([1.0 - 2 * k, 0, 0])) for k, arm in enumerate(arms)],
    ]
    lo = np.r_[[-0.092, -0.25, 0.197, -0.35, -0.32], *[arm.limits[0] for arm in arms]]
    hi = np.r_[[0.092, 0.12, 0.37, 0.35, 0.32], *[arm.limits[1] for arm in arms]]
    fit = least_squares(residual, initial, bounds=(lo, hi), max_nfev=110)
    xyz = fit.x[:3]
    axis = np.r_[1.0, fit.x[3:5]]
    axis /= np.linalg.norm(axis)
    horizontal = np.cross([0, 0, 1], axis)
    horizontal /= np.linalg.norm(horizontal)
    jaw = Rotation.from_rotvec(np.deg2rad(roll) * axis).apply(horizontal)
    jaws = (jaw, np.cross(axis, jaw))
    poses = fit.x[5:].reshape(2, 6) if reference else (None, None)
    return xyz, axis, jaws, poses
