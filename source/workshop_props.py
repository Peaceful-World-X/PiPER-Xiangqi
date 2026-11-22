"""Place the reference workshop's photographed Poly Haven props in metres."""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation


def scanned_props(scene, path):
    def place(name, asset, position, scale, yaw=0):
        root = scene.group(path + "/" + name, position,
                           Rotation.from_euler("z", yaw, degrees=True))
        root.AddScaleOp().Set(Gf.Vec3f(scale))
        model = UsdGeom.Xform.Define(scene.s, root.GetPath().AppendChild("Model"))
        model.GetPrim().GetReferences().AddReference(
            f"./assets/polyhaven/{asset}/{asset}_1k.usdc"
        )
        return model.GetPrim()

    tabletop = -.015
    place("ToolTray", "metal_toolbox", (.425, .407, tabletop), .352)
    place("TapeMeasure", "measuring_tape_01", (.235, .406, tabletop), .655, 90)
    # The open end is +Y in the scan. The ring end points towards the board.
    place("Spanner", "combination_wrench", (-.456, .370, tabletop+.0027), .442, 63)
    # The GALLER stamp is on the short faces of the scanned crate.
    crates = [
        place("CrateOnShelf", "plastic_crate_01", (-1.299, .350, -.1295), .85, 3),
        place("FloorCrate", "plastic_crate_01", (-1.426, -.235, -.770), .85, -8),
    ]
    for crate, finish in zip(crates, ("shelf", "floor")):
        # Faded red plastic retains the photographed scratches and lettering.
        for relative in ("Image_Texture", "NodeGraphs/bnode__Image_Texture_Color"):
            shader = UsdShade.Shader.Get(scene.s, crate.GetPath().AppendPath(
                "_materials/plastic_crate_01/" + relative))
            shader.GetInput("file").Set(Sdf.AssetPath(f"./assets/surfaces/galler_{finish}_print.jpg"))
    place("OilCan", "small_oil_can_01", (.978, .504, .1355), .63, 5)

    # Keep the collection's shared material bindings while arranging its
    # individual items. Its distributed model lays the pens flat beside the cup.
    model = place("Cup", "stationery_supplies", (-.225, .440, tabletop), 1.0)
    items = {
        "pencilcup": ((-.097, 0, .074), .84, (0, 0, 1)),
        "pen_blue": ((.146, 0, -.034), .90, (-.16, .03, .987)),
        "pen_fancy": ((.137, 0, -.065), 1.0, (.055, .12, .991)),
        "pencil_new_b": ((.150, 0, .041), .92, (.16, -.055, .985)),
    }
    bases = {"pen_blue": (-.009, -.004, .004),
             "pen_fancy": (.002, .006, .005),
             "pencil_new_b": (.008, -.006, .004)}
    for child in model.GetChildren():
        if not child.IsA(UsdGeom.Xform):
            continue
        name = child.GetName().removeprefix("stationery_supplies_")
        if name not in items:
            child.SetActive(False)
            continue
        shift, scale, direction = items[name]
        transform = UsdGeom.Xformable(child)
        mat = np.array(transform.GetLocalTransformation())
        rotation = Rotation.identity()
        if name in bases:
            rotation, _ = Rotation.align_vectors([direction], [[1, 0, 0]])
        place_mat = np.eye(4)
        place_mat[:3, :3] = scale * rotation.as_matrix().T
        place_mat[3, :3] = np.array(shift) @ place_mat[:3, :3] + bases.get(name, (0, 0, 0))
        transform.MakeMatrixXform().Set(Gf.Matrix4d((mat @ place_mat).tolist()))
