"""Rounded moulded lid and recessed channels on the floor tool case."""

import numpy as np
from pxr import UsdGeom, Vt


def floor_tool_case(g, path):
    g.material(
        "CaseBody", (0.03, 0.03, 0.028), 0.76, 0, "./assets/surfaces/case_shell.jpg"
    )
    g.material(
        "CaseLid", (0.266, 0.079, 0.080), 0.66, 0, "./assets/surfaces/case_lid.jpg"
    )
    g.material("CaseLip", (0.285, 0.092, 0.090), 0.59, 0)
    g.material("CaseClip", (0.53, 0.44, 0.31), 0.62, 0)
    body = g.box(
        path + "/Base", (0, 0, 0.090), (0.280, 0.201, 0.180), "CaseBody", 0.012
    )
    st = UsdGeom.PrimvarsAPI(body).GetPrimvar("st")
    uv = np.array(st.Get())
    uv[:, 1] *= 0.5
    uv[32:48, 1] += 0.5  # Only the front of the box carries the worn maker stamp.
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype("float32")))
    g.box(path + "/Lid", (0, 0, 0.188), (0.291, 0.210, 0.017), "CaseLip", 0.008)
    y, x = np.mgrid[-0.101:0.101:133j, -1.0:1.0:189j]
    inset = 0.010 - np.sqrt(
        np.maximum(0, 0.010**2 - np.maximum(0, abs(y) - 0.091) ** 2)
    )
    x = x * (0.142 - inset)
    d = (
        np.hypot(np.maximum(abs(x) - 0.128, 0), np.maximum(abs(y) - 0.087, 0))
        + np.minimum(np.maximum(abs(x) - 0.128, abs(y) - 0.087), 0)
        - 0.009
    )
    z = 0.1990 + 0.0018 * np.exp(-(((d + 0.002) / 0.0025) ** 2))
    for station in np.linspace(-0.110, 0.110, 8):
        dist = np.hypot(x - station, np.maximum(abs(y) - 0.071, 0))
        z -= 0.0019 * np.exp(-((dist / 0.00245) ** 2))
    rows, cols = x.shape
    points = np.dstack([x, y, z])
    normals = np.cross(np.gradient(points, axis=1), np.gradient(points, axis=0))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    start = np.arange(rows * cols).reshape(rows, cols)[:-1, :-1].ravel()
    faces = np.column_stack([start, start + 1, start + cols + 1, start + cols])
    uv = np.column_stack([0.5 + x.ravel() / 0.284, 0.5 + y.ravel() / 0.202])
    g.mesh(
        path + "/MouldedPanel",
        points.reshape(-1, 3),
        faces,
        "CaseLid",
        normals.reshape(-1, 3),
        uv,
    )
    for i, side in enumerate((-1, 1)):
        g.box(
            path + f"/Catch{i}",
            (side * 0.144, -0.008, 0.185),
            (0.005, 0.050, 0.020),
            "CaseClip",
            0.002,
        )
        g.box(
            path + f"/CatchTop{i}",
            (side * 0.141, -0.008, 0.1975),
            (0.008, 0.050, 0.004),
            "CaseClip",
            0.0013,
        )
