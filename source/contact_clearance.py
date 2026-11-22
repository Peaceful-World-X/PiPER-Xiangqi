"""Measure the CAD gripper clearance above the board and tabletop."""

import numpy as np
from scipy.spatial import ConvexHull


class GripperClearance:
    def __init__(self, model, data, base, Z, board_height):
        self.d = data
        self.base = base
        self.Z = Z
        self.h = board_height
        self.parts = []
        for i in range(model.ngeom):
            body = model.geom_bodyid[i]
            if model.geom_group[i] != 2 or model.body(body).name not in (
                "link5",
                "link6",
                "link7",
                "link8",
            ):
                continue
            k = model.geom_dataid[i]
            vs = model.mesh_vert[
                model.mesh_vertadr[k] : model.mesh_vertadr[k] + model.mesh_vertnum[k]
            ]
            points = vs[ConvexHull(vs).vertices]
            self.parts.append((i, points))

    def deficit(self):
        need = 0.0
        for i, v in self.parts:
            world = (
                self.Z.apply(
                    v @ self.d.geom_xmat[i].reshape(3, 3).T + self.d.geom_xpos[i]
                )
                + self.base
            )
            x, y, z = world.T
            level = np.where(
                (abs(x) < 0.226) & (abs(y) < self.h / 2 + 0.001), 0.00012, -0.0148
            )
            need = max(need, float(np.max(level + 0.0003 - z)))
        return need
