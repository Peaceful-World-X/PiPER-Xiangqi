"""Author the editable room, native USD animation and the measured shot camera."""

from pathlib import Path
import argparse
import numpy as np, mujoco as mj
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from scipy.spatial.transform import Rotation
from geometry import Scene, quat
from shot import FPS, LAST, camera_samples
from robots import build_model
from rig_motion import plan_motion, BASE_ROTATION, TOOL_LENGTH, PIECE_RADIUS, PIECE_HEIGHT
from workshop import furnish, lights, reference_surfaces

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
cam = np.load(A / "tracking.npz")
motion = np.load(A / "motion.npz")
parser = argparse.ArgumentParser()
parser.add_argument("--variant", "--lighting", dest="variant", choices=("v4", "reference"), default="v4",
                    help="v4 release or the reference-matched motion and lighting")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
reference = args.variant == "reference"
scene_name = "Workshop-shadows.usdc" if reference else "Workshop.usdc"
output = (args.output or ROOT / scene_name).resolve()
stage = Usd.Stage.CreateNew(str(output))
UsdGeom.SetStageUpAxis(stage, "Z")
UsdGeom.SetStageMetersPerUnit(stage, 1)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(LAST)
stage.SetTimeCodesPerSecond(FPS)
stage.SetFramesPerSecond(FPS)
g = Scene(stage)
root = g.group("/World")
stage.SetDefaultPrim(root.GetPrim())
if reference:
    from reference_plan import reference_motion
    motion = reference_motion(motion)
    # Joint fit of front cable connectors in eight board-registered views.
    motion["base"] = np.array([[-.31125, -.196, .001], [.31214, -.196, .001]])
furnish(g, float(cam["aspect"]) * .45, motion["base"])
if reference:
    reference_surfaces(g)
lights(g, reference)
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.CreateHorizontalApertureAttr(36.0)
camera.CreateVerticalApertureAttr(20.25)
camera.CreateFocalLengthAttr(float(cam["K"][0, 0]) * 36 / 960)
camera.CreateClippingRangeAttr((0.015, 100.0))
move = camera.AddTranslateOp()
ori = camera.AddOrientOp()
model = mj.MjModel.from_xml_path(str(A / "agilex_piper/piper.xml"))
qs, gaps, plan, source_time = plan_motion(model, motion, cam, reference=reference)
source_attr = camera.GetPrim().CreateAttribute("shot:referenceSeconds", Sdf.ValueTypeNames.Double)
eyes, rotations = camera_samples(cam, source_time)
for i, (pos, rotation) in enumerate(zip(eyes, rotations)):
    q = quat(Rotation.from_rotvec(rotation))
    if i == 0:
        move.Set(Gf.Vec3d(*pos))
        ori.Set(q)
    source_attr.Set(float(source_time[i]), i)
    move.Set(Gf.Vec3d(*pos), i)
    ori.Set(q, i)
stage.GetRootLayer().customLayerData = {
    "cameraSettings": {"boundCamera": "/World/Camera"}
}
# The resting quadruped uses the supplier geometry in a folded pose.
dog = mj.MjModel.from_xml_path(str(A / "unitree_go2/go2.xml"))
dq = dog.qpos0.copy()
dq[:7] = [0, 0, 0.142, 1, 0, 0, 0]
for j in range(12):
    name = dog.joint(j + 1).name
    dq[7 + j] = (
        0.06 * (1 if name[1] == "L" else -1)
        if "hip" in name
        else (1.22 if "thigh" in name else -2.62)
    )
g.group("/World/Go2", (-0.10, 0.16, -0.77), Rotation.from_euler("z", -6, degrees=True))
build_model(g, "/World/Go2", dog, dq)
data = mj.MjData(model)
frames = np.arange(len(qs[0]))
tips = []
axes = []
for k, side in enumerate(["Left", "Right"]):
    path = "/World/" + side + "PiPER"
    base = motion["base"][k]
    g.group(path, base, BASE_ROTATION)
    joints = build_model(g, path, model, reference=reference)
    tt = []
    aa = []
    for f, q, opening in zip(frames, qs[k], gaps[k]):
        data.qpos[:] = [*q, opening, -opening]
        mj.mj_kinematics(model, data)
        for j, op in joints.items():
            displacement = float(data.qpos[model.jnt_qposadr[j]])
            if model.jnt_type[j] == mj.mjtJoint.mjJNT_SLIDE:
                value = Gf.Vec3d(*map(float, model.jnt_axis[j] * displacement))
            else:
                value = float(np.rad2deg(displacement) * model.jnt_axis[j, 2])
            if f == 0:
                op.Set(value)
            op.Set(value, int(f))
        wrist = data.body("link6")
        direction = BASE_ROTATION.apply(wrist.xmat.reshape(3, 3)[:, 2])
        tt.append(base + BASE_ROTATION.apply(wrist.xpos) + direction * TOOL_LENGTH)
        aa.append(BASE_ROTATION.as_matrix() @ wrist.xmat.reshape(3, 3))
    tips.append(np.array(tt))
    axes.append(np.array(aa))
tips = np.array(tips)
axes = np.array(axes)


g.material("PieceBack", (.48, .31, .16), .62, 0, "./assets/surfaces/board_edge.png")
g.material("TurnedPieceWood", (.300, .157, .073), .55, 0)

for j, event in enumerate(plan):
    donor, receiver, src = map(int, event[:3])
    take, handover, drop = (round(t * FPS) if t >= 0 else -1 for t in event[3:])
    path = f"/World/Pieces/Piece_{j:02}"
    piece = g.group(path)
    g.material(
        "Face" + str(j),
        (0.5, 0.3, 0.14),
        0.58,
        0,
        f"./assets/surfaces/piece_{j:02}.png",
    )
    # The measured photographs already contain individual wood and ink variation.
    # Restore print contrast in linear colour before lighting the disc.
    face_image = UsdShade.Shader.Get(stage, f"/World/Materials/Face{j}/Image")
    face_image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.02, .74, .71, 1))
    face_image.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-.112, -.036, -.024, 0))
    g.disc(path, PIECE_RADIUS, PIECE_HEIGHT, "TurnedPieceWood", "Face" + str(j),
           centred=True, back="PieceBack")
    for name, value in (("donor", donor), ("receiver", receiver), ("take", take),
                        ("handover", handover), ("drop", drop), ("source", src)):
        piece.GetPrim().CreateAttribute("grasp:" + name, Sdf.ValueTypeNames.Int).Set(value)
    tr = piece.AddTranslateOp()
    orient = piece.AddOrientOp()
    phases = [(donor, take, drop)] if handover < 0 else [
        (donor, take, handover), (receiver, handover, drop)
    ]
    net = Rotation.identity()
    for owner, begin, finish in phases:
        net = Rotation.from_matrix(axes[owner, finish] @ axes[owner, begin].T) * net
    # A handover flips the token, as the receiver approaches its opposite face.
    # Choose its initial heading so the printed face ends upright on the board.
    normal = np.array([0, 0, 1 if handover < 0 else -1])
    right = net.inv().apply([1, 0, 0])
    right[2] = 0
    right /= np.linalg.norm(right)
    attitude = Rotation.from_matrix(np.column_stack((right, np.cross(normal, right), normal)))
    centre = np.r_[motion["starts"][src], motion["startz"][src] + PIECE_HEIGHT / 2]
    def set_pose(frame, pos, rotation):
        tr.Set(Gf.Vec3d(*pos), frame)
        orient.Set(quat(rotation), frame)
    set_pose(Usd.TimeCode.Default(), centre, attitude)
    set_pose(0, centre, attitude)
    for owner, begin, finish in phases:
        initial_wrist = Rotation.from_matrix(axes[owner, begin])
        local_rotation = initial_wrist.inv() * attitude
        local_offset = initial_wrist.inv().apply(centre - tips[owner, begin])
        for f in range(begin, finish + 1):
            wrist_rotation = Rotation.from_matrix(axes[owner, f])
            centre = tips[owner, f] + wrist_rotation.apply(local_offset)
            attitude = wrist_rotation * local_rotation
            set_pose(f, centre, attitude)
    # Let the two long-reach placements settle from their lower rim as the jaws release.
    x_axis = attitude.as_matrix()[:, 0]
    flat = Rotation.from_euler("z", np.arctan2(x_axis[1], x_axis[0]))
    relative = (flat * attitude.inv()).as_rotvec()
    placed = centre.copy()
    settle_steps = round(.16 * FPS)
    release_frame = drop + 2
    set_pose(release_frame, centre, attitude)
    for k in range(1, settle_steps + 1):
        t = k / settle_steps
        blend = t*t*(3-2*t)
        rot = Rotation.from_rotvec(relative * blend) * attitude
        tilt_cos = np.clip(rot.as_matrix()[2, 2], 0, 1)
        placed[2] = .00013 + PIECE_HEIGHT / 2 * tilt_cos + .915 * PIECE_RADIUS * np.sqrt(1-tilt_cos**2)
        set_pose(release_frame + k, placed, rot)
    set_pose(LAST, placed, flat)
stage.GetRootLayer().Save()
print("Saved", output, flush=True)
