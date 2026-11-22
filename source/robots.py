"""Convert the articulated visual meshes to time-sampled native USD links."""

import numpy as np
import mujoco as mj
from pxr import UsdShade
from scipy.spatial.transform import Rotation


def rot(q):
    return Rotation.from_quat(np.roll(q, -1))


PALETTE = {
    "light_gray_mat": (0.64, 0.65, 0.64),
    "dark_gray_mat": (0.037, 0.039, 0.04),
    "darker_gray_mat": (0.032, 0.033, 0.034),
    "red_mat": (0.40, 0.021, 0.018),
    "black_mat": (0.014, 0.015, 0.016),
    "metal": (0.39, 0.42, 0.44),
    "black": (0.013, 0.017, 0.018),
    "white": (0.48, 0.50, 0.51),
    "gray": (0.27, 0.29, 0.31),
}


def build_model(scene, root, model, static_q=None, reference=False):
    d = mj.MjData(model)
    if static_q is not None:
        d.qpos[:] = static_q
    mj.mj_forward(model, d)
    for k, col in PALETTE.items():
        if k not in scene.materials:
            scene.material(
                k,
                col,
                0.30 if k == "metal" else 0.45,
                0.65 if k == "metal" else 0.12,
            )
    # The PiPER reference has continuous satin highlights across the painted panels
    # and machined sleeves. Use the original smooth CAD normals and solid finishes.
    if static_q is None and "PiPERShell" not in scene.materials:
        if reference:
            scene.material("PiPERShell", (.245, .255, .267), .43, .12)
            scene.material("PiPERBarrel", (.049, .053, .056), .53, .55)
            scene.material("PiPERSleeve", (.042, .044, .045), .48, .34)
            scene.material("PiPERWristCap", (.164, .168, .170), .56, .12)
        else:
            scene.material("PiPERShell", (.351, .370, .391), .32, .12)
            scene.material("PiPERBarrel", (.096, .105, .112), .255, .65)
            scene.material("PiPERSleeve", (.061, .063, .063), .36, .34)
    paths = {0: root}
    ops = {}
    for i in range(1, model.nbody):
        parent = model.body_parentid[i]
        name = model.body(i).name
        if static_q is not None:
            paths[i] = root + "/" + name
            scene.group(paths[i], d.xpos[i], rot(d.xquat[i]))
            continue
        paths[i] = paths[parent] + "/" + name
        body = scene.group(paths[i], model.body_pos[i], rot(model.body_quat[i]))
        if not model.body_jntnum[i]:
            continue
        joint = model.body_jntadr[i]
        axis = model.jnt_axis[joint]
        if (
            model.body_jntnum[i] != 1
            or np.any(model.jnt_pos[joint])
            or not np.allclose(abs(axis), [0, 0, 1])
        ):
            raise ValueError(f"Unexpected PiPER pivot or axis on {name}")
        if model.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE:
            op = body.AddRotateZOp(opSuffix="joint")
        elif model.jnt_type[joint] == mj.mjtJoint.mjJNT_SLIDE:
            op = body.AddTranslateOp(opSuffix="jawSlide")
        else:
            raise ValueError(f"Unexpected PiPER joint type on {name}")
        ops[joint] = op
    for i in range(model.ngeom):
        if model.geom_group[i] != 2:
            continue
        m = model.geom_dataid[i]
        name = model.mesh(m).name
        body = model.geom_bodyid[i]
        vi = model.mesh_vertadr[m]
        fi = model.mesh_faceadr[m]
        ni = model.mesh_normaladr[m]
        nface = model.mesh_facenum[m]
        vs = model.mesh_vert[vi : vi + model.mesh_vertnum[m]]
        faces = model.mesh_face[fi : fi + nface]
        normals = model.mesh_normal[ni + model.mesh_facenormal[fi : fi + nface]]
        mat = model.mat(model.geom_matid[i]).name
        if name in ("link2_0", "link3_14"):
            mat = "white_mat"
        if name == "link2_31":
            mat = "darker_gray_mat"
        if static_q is None:
            if reference and name == "link5_6":
                mat = "PiPERWristCap"
            elif mat == "white_mat":
                mat = "PiPERShell"
            elif name in ("link3_18", "link3_19", "link3_20"):
                mat = "PiPERBarrel"
            elif mat == "gray_mat":
                mat = "PiPERSleeve"
        ob = scene.mesh(
            paths[body] + "/Geom_" + str(i),
            vs,
            faces,
            mat,
            normals,
        )
        scene.transform(ob, model.geom_pos[i], rot(model.geom_quat[i]))
        if model.names.startswith(b"go2") and name == "base_3":
            # The supplier mesh combines the side wordmark, nose stripes and
            # opposite-side branding. Select the three Go2 glyphs in body space.
            body_points = rot(model.geom_quat[i]).apply(vs) + model.geom_pos[i]
            mid = body_points[faces].mean(axis=1)
            letters = np.flatnonzero((mid[:, 0] < .08) & (mid[:, 1] < -.06))
            scene.material("Go2Ink", (.012, .013, .014), .67)
            subset = UsdShade.MaterialBindingAPI(ob).CreateMaterialBindSubset(
                "Go2Lettering", letters.tolist())
            scene.bind(subset, "Go2Ink")
    if static_q is None:
        decorate_arm(
            scene, paths[model.body("link2").id], paths[model.body("link3").id]
        )
    return ops


def decorate_arm(scene, upper, fore):
    for name, image in [
        ("ArmPrint", "arm_nameplate"),
        ("ForearmPrint", "forearm_nameplate"),
        ("JointPrint", "joint_badge"),
    ]:
        if name not in scene.materials:
            scene.material(
                name, (0.04, 0.04, 0.04), 0.37, 0.04, f"./assets/surfaces/{image}.png"
            )
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    for sign, side in [(1, "A"), (-1, "B")]:
        for body, mat, center, length, height, angle, depth in [
            (upper, "ArmPrint", [0.1379, -0.0053], 0.101, 0.017, -0.087, 0.02055),
            (fore, "ForearmPrint", [-0.0945, 0.0402], 0.051, 0.0086, -0.19, 0.01955),
        ]:
            u = np.array([np.cos(angle), np.sin(angle)])
            v = np.array([-u[1], u[0]])
            xy = (
                center
                + sign * (uv[:, :1] - 0.5) * length * u
                + (uv[:, 1:] - 0.5) * height * v
            )
            z = np.full(4, depth)
            if body == fore:
                z += 0.0782 * (xy[:, 0] - center[0]) - 0.015 * (xy[:, 1] - center[1])
            scene.mesh(
                body + "/PrintedPanel" + side,
                np.c_[xy, sign * z],
                [[0, 1, 2, 3]],
                mat,
                uv=uv,
            )
        for index, (cx, cy) in enumerate([(0, 0), (0.2836, 0.0287)]):
            angles = np.arange(80) * 2 * np.pi / 80
            x = np.r_[0, np.cos(angles)]
            y = np.r_[0, np.sin(angles)]
            points = np.c_[
                cx + 0.0181 * sign * x, cy + 0.0181 * y, np.full(len(x), sign * 0.04748)
            ]
            faces = [[0, i + 1, (i + 1) % 80 + 1] for i in range(80)]
            scene.mesh(
                upper + f"/JointLegend{index}{side}",
                points,
                faces,
                "JointPrint",
                uv=np.c_[0.5 + 0.5 * x, 0.5 + 0.5 * y],
            )
