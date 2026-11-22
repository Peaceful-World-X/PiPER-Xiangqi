"""Small USD geometry and Preview Surface authoring helpers. Units are metres."""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
from scipy.spatial.transform import Rotation


def quat(r):
    x, y, z, w = r.as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def v3(p):
    return Gf.Vec3f(*map(float, p))


class Scene:
    def __init__(self, stage):
        self.s = stage
        self.materials = {}

    def material(
        self,
        name,
        color,
        rough=0.58,
        metal=0,
        texture=None,
        surface_maps=None,
        coat=0.0,
        coat_roughness=0.18,
        normal_map=True,
    ):
        p = "/World/Materials/" + name
        m = UsdShade.Material.Define(self.s, p)
        sh = UsdShade.Shader.Define(self.s, p + "/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        if coat:
            sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(coat)
            sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
                coat_roughness
            )
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(v3(color))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        if texture:
            uv = UsdShade.Shader.Define(self.s, p + "/UV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(self.s, p + "/Image")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            sh.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
            if surface_maps:
                for channel, suffix, output in [
                    ("roughness", "rough", "r"),
                    ("normal", "normal", "rgb"),
                ]:
                    if channel == "normal" and not normal_map:
                        continue
                    node = UsdShade.Shader.Define(self.s, p + "/" + suffix)
                    node.CreateIdAttr("UsdUVTexture")
                    node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                        f"./assets/surfaces/{surface_maps}_{suffix}.png"
                    )
                    node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                        "raw"
                    )
                    node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                        uv.ConnectableAPI(), "result"
                    )
                    if channel == "normal":
                        node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                            (2, 2, 2, 1)
                        )
                        node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                            (-1, -1, -1, 0)
                        )
                        sh.CreateInput(channel, Sdf.ValueTypeNames.Normal3f)
                    sh.GetInput(channel).ConnectToSource(node.ConnectableAPI(), output)
        self.materials[name] = m

    def bind(self, p, mat):
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(self.materials[mat])

    def transform(self, prim, pos=None, rot=None, scale=None):
        if pos is not None:
            prim.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        if rot is not None:
            prim.AddOrientOp().Set(quat(rot))
        if scale is not None:
            prim.AddScaleOp().Set(v3(scale))
        return prim

    def group(self, path, pos=None, rot=None):
        return self.transform(UsdGeom.Xform.Define(self.s, path), pos, rot)

    def mesh(self, p, verts, faces, mat, normals=None, uv=None):
        v = np.ascontiguousarray(verts, dtype=np.float32)
        f = np.ascontiguousarray(faces, dtype=np.int32)
        prim = UsdGeom.Mesh.Define(self.s, p)
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
        prim.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(f), f.shape[1], np.int32))
        )
        prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
        prim.CreateSubdivisionSchemeAttr("none")
        prim.CreateExtentAttr([v3(v.min(0)), v3(v.max(0))])
        if normals is not None:
            n = np.ascontiguousarray(normals, dtype="float32").reshape(-1, 3)
            prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n))
            prim.SetNormalsInterpolation(
                "vertex" if len(n) == len(v) else "faceVarying"
            )
        if uv is not None:
            uv = np.asarray(uv, dtype="float32")
            interpolation = "faceVarying" if uv.ndim == 3 else "vertex"
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation
            ).Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv.reshape(-1, 2), dtype="float32")))
        self.bind(prim, mat)
        return prim

    def box(self, p, center, size, mat, rad=0.001, rot=None):
        # A bevel around each edge catches the window highlight at grazing angles.
        half = np.array(size) * 0.5
        r = min(rad, float(half.min()) * 0.8)
        points = []
        normals = []
        faces = []
        uv = []
        for axis, other1, other2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for side in (-1, 1):
                start = len(points)
                h1, h2 = half[other1], half[other2]
                for b in [-h2, -h2 + r, h2 - r, h2]:
                    for a in [-h1, -h1 + r, h1 - r, h1]:
                        v = np.zeros(3)
                        v[axis] = half[axis] * side
                        v[other1] = a
                        v[other2] = b
                        inner = np.clip(v, -half + r, half - r)
                        n = v - inner
                        n /= np.linalg.norm(n)
                        points.append(inner + n * r)
                        normals.append(n)
                        # Horizontal direction follows the grain on shelves and drawer faces.
                        uv.append(
                            ((a / h1 + 1) * 0.5, (b / h2 + 1) * 0.5)
                            if axis != 1
                            else ((b / h2 + 1) * 0.5, (a / h1 + 1) * 0.5)
                        )
                for row in range(3):
                    for col in range(3):
                        j = start + 4 * row + col
                        face = [j, j + 1, j + 5, j + 4]
                        faces.append(face if side > 0 else face[::-1])
        if mat == "Concrete":
            uv = np.array(uv) * [11, 9]
        if mat == "Plaster":
            uv = np.array(uv) * [2, 4]
        return self.transform(
            self.mesh(p, points, faces, mat, normals, uv), center, rot
        )

    def tube(self, p, a, b, r, mat):
        a, b = np.array(a), np.array(b)
        d = b - a
        n = np.linalg.norm(d)
        prim = UsdGeom.Cylinder.Define(self.s, p)
        prim.CreateRadiusAttr(r)
        prim.CreateHeightAttr(float(n))
        self.bind(prim, mat)
        axis = np.cross([0.0, 0.0, 1.0], d)
        ang = np.arccos(np.clip(d[2] / n, -1, 1))
        rot = (
            Rotation.from_rotvec(axis * ang / np.linalg.norm(axis))
            if np.linalg.norm(axis) > 1e-8
            else Rotation.from_euler("x", ang)
        )
        return self.transform(prim, (a + b) / 2, rot)

    def wire(self, p, pts, r, mat):
        pts = np.array(pts)
        curve = UsdGeom.BasisCurves.Define(self.s, p)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype("float32")))
        curve.CreateWidthsAttr([2 * r])
        curve.SetWidthsInterpolation("constant")
        self.bind(curve, mat)
        return curve

    def disc(self, p, r, thick, mat, face, *, centred=False, back=None):
        # Turned wooden disc with a rounded lip and a separately mapped top face.
        v = []
        n = []
        uv = []
        f = []
        N = 64
        for rr, z, nr, nz in [
            (r * 0.91, 0, 0.35, -0.94),
            (r * 0.985, 0.001, 0.85, -0.53),
            (r, 0.0025, 1, 0),
            (r, thick - 0.002, 1, 0),
            (r * 0.975, thick - 0.0007, 0.75, 0.66),
            (r * 0.915, thick, 0.12, 0.99),
        ]:
            for i in range(N):
                x, y = np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)
                v.append((rr * x, rr * y, z))
                n.append((nr * x, nr * y, nz))
                uv.append((0.5 + 0.5 * x, 0.5 + 0.5 * y))
        for j in range(5):
            for i in range(N):
                a = j * N + i
                b = j * N + (i + 1) % N
                f.append((a, b, b + N, a + N))
        offset = thick / 2 if centred else 0.0
        v = np.asarray(v)
        v[:, 2] -= offset
        self.mesh(p + "/Edge", v, f, mat, n, uv)
        ang = np.arange(N) * 2 * np.pi / N
        x = np.cos(ang)
        y = np.sin(ang)
        v = np.column_stack(
            [
                np.r_[0, x * r * 0.915],
                np.r_[0, y * r * 0.915],
                np.full(N + 1, thick - offset + 0.00003),
            ]
        )
        self.mesh(
            p + "/Face",
            v,
            [[0, 1 + i, 1 + (i + 1) % N] for i in range(N)],
            face,
            None,
            np.c_[np.r_[0.5, 0.5 + 0.5 * x], np.r_[0.5, 0.5 + 0.5 * y]],
        )
        if back:
            v[:, 2] = -offset
            self.mesh(
                p + "/Back", v, [[0, 1 + (i + 1) % N, 1 + i] for i in range(N)],
                back, uv=np.c_[np.r_[0.5, 0.5 + 0.5 * x], np.r_[0.5, 0.5 + 0.5 * y]],
            )
