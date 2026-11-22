"""Choose the final wrist roll using the two CAD finger pads and the settled discs."""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


def plan_approach(arm, motion, index, event, xyz, axis, base_rotation):
    plan = motion["plan"]
    clock = plan[index, 3 if event == "take" else 5]
    idle = (plan[:, 3] > clock) | (plan[:, 5] < clock)
    idle[index] = False
    ids = plan[:, 2].astype(int)
    scattered = np.column_stack((motion["starts"][ids], motion["startz"][ids] + 0.0054))
    centres = np.where((plan[:, 5] < clock)[:, None], motion["dest"], scattered)
    nearby = centres[idle & (np.linalg.norm(centres[:, :2] - xyz[:2], axis=1) < 0.074)]
    if not len(nearby):
        return None, False
    seed = arm.solve(xyz, axis)
    _, basis = arm.fk(seed)
    # Evaluate at full lateral opening, partial closure and holding width.
    # Upper gripper surfaces cannot enter the low slab occupied by a flat disc.
    q = arm.data.qpos
    candidates = np.deg2rad([0, *[v for n in range(6, 91, 6) for v in (n, -n)]])
    candidates = candidates[
        (seed[5] + candidates >= arm.limits[0, 5])
        & (seed[5] + candidates <= arm.limits[1, 5])
    ]
    pads = []
    for aperture in (0.01815, 0.021, 0.024):
        q[:] = [*seed, aperture, -aperture]
        mj.mj_kinematics(arm.model, arm.data)
        for i, body in enumerate(arm.model.geom_bodyid):
            if arm.model.geom_group[i] != 2 or arm.model.body(body).name not in (
                "link7",
                "link8",
            ):
                continue
            mesh = arm.model.geom_dataid[i]
            first = arm.model.mesh_vertadr[mesh]
            v = arm.model.mesh_vert[first : first + arm.model.mesh_vertnum[mesh]]
            world = (
                base_rotation.apply(
                    v @ arm.data.geom_xmat[i].reshape(3, 3).T + arm.data.geom_xpos[i]
                )
                + arm.base
            )
            pads.append((world - xyz) @ basis)
    pads = np.vstack(pads)
    best = None
    for angle in candidates:
        rot = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        points = pads @ rot.T @ basis.T + xyz
        margin = 0.012
        for centre in nearby:
            slab = points[
                (points[:, 2] < centre[2] + 0.0056 + 0.0025)
                & (points[:, 2] > centre[2] - 0.0056 - 0.003)
            ]
            if len(slab):
                margin = min(
                    margin,
                    np.linalg.norm(slab[:, :2] - centre[:2], axis=1).min() - 0.0182,
                )
        score = min(margin, 0.0038) - 0.0002 * abs(angle)
        if best is None or score > best[0]:
            best = (score, angle, basis @ rot[:, 1])
    return best[2], True
