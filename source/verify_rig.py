"""Check both USD rigs, including between rendered frames, against the PiPER MJCF."""

from shot import FPS, FRAME_COUNT
import argparse
from pathlib import Path
import mujoco as mj
import numpy as np
from pxr import Usd, UsdGeom, UsdUtils
from contact_clearance import GripperClearance
from rig_motion import BASE_ROTATION, VELOCITY, ACCELERATION

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", nargs="?", type=Path, default=root / "Workshop.usdc")
path = parser.parse_args().stage.resolve()
stage = Usd.Stage.Open(str(path))
model = mj.MjModel.from_xml_path(str(root / "assets/agilex_piper/piper.xml"))
geometry = np.load(root / "assets/tracking.npz")
_, _, missing = UsdUtils.ComputeAllDependencies(str(path))
assert not missing, missing
frames = range(int(stage.GetEndTimeCode()) + 1)
assert len(frames) == FRAME_COUNT
assert stage.GetTimeCodesPerSecond() == stage.GetFramesPerSecond() == FPS
# Floor props are static throughout the shot. Check the stage meshes so a future
# change in shelf depth or carton position cannot reintroduce the intersection.
bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
transform_cache = UsdGeom.XformCache()


def workshop_prim(name):
    prim = stage.GetPrimAtPath("/World/Workshop/" + name)
    assert prim.IsValid(), name
    return prim


def limits(name):
    box = bounds.ComputeWorldBound(workshop_prim(name)).ComputeAlignedRange()
    return np.array([box.GetMin(), box.GetMax()])


def footprint(name):
    from scipy.spatial import ConvexHull
    points = []
    for prim in Usd.PrimRange(workshop_prim(name)):
        if prim.IsA(UsdGeom.Mesh):
            local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            matrix = np.array(transform_cache.GetLocalToWorldTransform(prim))
            points.append((local @ matrix[:3, :3] + matrix[3, :3])[:, :2])
    xy = np.vstack(points)
    return xy[ConvexHull(xy).vertices]


carton = limits("ParcelOnFloor")
rack = limits("RightRack")
rack_gap = rack[0, 1] - carton[1, 1]
assert rack_gap > 0, "The floor carton overlaps the right rack depth range"
assert abs(carton[0, 2] - limits("Floor")[1, 2]) < .0002
assert abs(limits("ParcelOnShelf")[0, 2] - limits("RightRack/Shelf1")[1, 2]) < .0002
carton_xy, case_xy = footprint("ParcelOnFloor"), footprint("ToolCase")
edges = np.vstack([np.roll(poly, -1, axis=0) - poly for poly in (carton_xy, case_xy)])
axes = np.column_stack((-edges[:, 1], edges[:, 0]))
axes /= np.linalg.norm(axes, axis=1)[:, None]
box_proj, case_proj = carton_xy @ axes.T, case_xy @ axes.T
case_gap = max((box_proj.min(0) - case_proj.max(0)).max(), (case_proj.min(0) - box_proj.max(0)).max())
assert case_gap > 0, "The floor carton overlaps the tool case footprint"
print(f"Floor carton clearance: rack at least {1000*rack_gap:.1f} mm, "
      f"tool case at least {1000*case_gap:.1f} mm; floor and shelf contacts verified.", flush=True)



for side in ("Left", "Right"):
    arm = stage.GetPrimAtPath("/World/" + side + "PiPER")
    base = np.array(
        UsdGeom.Xformable(arm).GetLocalTransformation().ExtractTranslation()
    )
    data = mj.MjData(model)
    clearance = GripperClearance(
        model, data, base, BASE_ROTATION, 0.45 * float(geometry["aspect"])
    )
    paths = {0: str(arm.GetPath())}
    bodies = [None]
    channels = []
    for body_id in range(1, model.nbody):
        paths[body_id] = (
            paths[model.body_parentid[body_id]] + "/" + model.body(body_id).name
        )
        body = stage.GetPrimAtPath(paths[body_id])
        assert body.IsValid(), paths[body_id]
        bodies.append(body)
        ops = UsdGeom.Xformable(body).GetOrderedXformOps()
        assert not ops[0].GetAttr().ValueMightBeTimeVarying()
        assert not ops[1].GetAttr().ValueMightBeTimeVarying()
        if model.body_jntnum[body_id]:
            assert len(ops) == 3 and ops[2].GetNumTimeSamples() == len(frames)
            channels.append(ops[2])
    max_pos = max_rot = max_deficit = 0.0
    angles = []
    for f2 in range(2 * len(frames) - 1):
        time = Usd.TimeCode(f2 * 0.5)
        for j, channel in enumerate(channels):
            value = channel.Get(time)
            if model.jnt_type[j] == mj.mjtJoint.mjJNT_SLIDE:
                data.qpos[j] = np.dot(model.jnt_axis[j], value)
            else:
                data.qpos[j] = np.deg2rad(value) * model.jnt_axis[j, 2]
        assert np.all(data.qpos >= model.jnt_range[:, 0] - 1e-6) and np.all(
            data.qpos <= model.jnt_range[:, 1] + 1e-6
        )
        mj.mj_kinematics(model, data)
        max_deficit = max(max_deficit, clearance.deficit())
        cache = UsdGeom.XformCache(time)
        for body_id in range(1, model.nbody):
            mat = np.array(cache.GetLocalToWorldTransform(bodies[body_id]))
            position = base + BASE_ROTATION.apply(data.xpos[body_id])
            direction = BASE_ROTATION.as_matrix() @ data.xmat[body_id].reshape(3, 3)
            max_pos = max(max_pos, np.linalg.norm(mat[3, :3] - position))
            max_rot = max(max_rot, np.max(abs(mat[:3, :3].T - direction)))
        if f2 % 2 == 0:
            angles.append(data.qpos[:6].copy())
    v = np.abs(np.diff(angles, axis=0) * FPS).max(0)
    acc = np.abs(np.diff(angles, n=2, axis=0) * FPS**2).max(0)
    print(
        side,
        "max origin mismatch (mm)",
        round(1000 * max_pos, 7),
        "max rotation matrix error",
        max_rot,
        "max support deficit incl 0.3 mm margin (mm)",
        round(1000 * max_deficit, 4),
        "peak deg/s",
        np.round(np.rad2deg(v), 1),
        "peak deg/s^2",
        np.round(np.rad2deg(acc), 1),
        flush=True,
    )
    assert max_pos < 0.00001 and max_rot < 1e-5 and max_deficit < 0.0003
    assert np.all(v < VELOCITY + 0.001) and np.all(acc < ACCELERATION + 0.01)
assert len(stage.GetPrimAtPath("/World/Pieces").GetChildren()) == 32
from grasp_check import verify_grasps
verify_grasps(stage)
from nearby_pieces import verify_neighbours
verify_neighbours(stage)
print(f"Rig check passed for {FRAME_COUNT} frames and {FRAME_COUNT - 1} midpoints.")
