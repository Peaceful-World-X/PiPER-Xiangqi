"""Check CAD finger triangles against the closed convex rim of every other token."""

import numpy as np
from pxr import UsdGeom
from scipy.spatial import ConvexHull


def intersects(points, faces, planes, extent, tolerance=0.000015):
    # Piece-local box first, then clip each remaining triangle against the
    # measured bevel planes. This finds edge crossings even with no vertex inside.
    tri = points[faces]
    keep = ((tri.min(axis=1) < extent) & (tri.max(axis=1) > -extent)).all(axis=1)
    tri = tri[keep]
    if not len(tri):
        return False
    distances = tri @ planes[:, :3].T + planes[:, 3] + tolerance
    possible = distances.min(axis=1).max(axis=1) <= 0
    for polygon, ds in zip(tri[possible], distances[possible]):
        if np.any(ds.max(axis=1) <= 0):
            return True
        for plane in planes[ds.max(axis=0) > 0]:
            signed = polygon @ plane[:3] + plane[3] + tolerance
            if np.all(signed > 0):
                break
            if np.all(signed <= 0):
                continue
            previous = np.roll(polygon, 1, axis=0)
            before = np.roll(signed, 1)
            clipped = []
            for a, b, da, db in zip(previous, polygon, before, signed):
                if (da <= 0) != (db <= 0):
                    clipped.append(a + (da / (da - db)) * (b - a))
                if db <= 0:
                    clipped.append(b)
            polygon = np.asarray(clipped)
        else:
            return True
    return False


def verify_neighbours(stage, stride=1, fail=True):
    pieces = list(stage.GetPrimAtPath("/World/Pieces").GetChildren())
    count = len(pieces)
    fingers = []
    for side in ("Left", "Right"):
        wrist = stage.GetPrimAtPath(
            f"/World/{side}PiPER/base_link/link1/link2/link3/link4/link5/link6"
        )
        for name in ("link7", "link8"):
            mesh = UsdGeom.Mesh(
                next(
                    p for p in wrist.GetChild(name).GetChildren() if p.IsA(UsdGeom.Mesh)
                )
            )
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            fingers.append(
                (mesh.GetPrim(), points, faces, points[ConvexHull(points).vertices])
            )
    # Only the explicitly scheduled piece may contact this hand on approach,
    # during a grasp and while opening or withdrawing from that same piece.
    legal = [[] for _ in range(2)]
    for j, piece in enumerate(pieces):
        d, r, t, h, p = (
            piece.GetAttribute("grasp:" + n).Get()
            for n in ("donor", "receiver", "take", "handover", "drop")
        )
        if h < 0:
            legal[d].append((j, t - 22, p + 22))
        else:
            legal[d].append((j, t - 22, h + 22))
            legal[r].append((j, h - 22, p + 22))
    mesh = UsdGeom.Mesh(pieces[0].GetChild("Edge"))
    rim = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    crown = np.asarray(
        UsdGeom.Mesh(pieces[0].GetChild("Face")).GetPointsAttr().Get(), dtype=float
    )
    shell = np.vstack([rim, crown])
    planes = np.unique(ConvexHull(shell).equations.round(9), axis=0)
    extent = np.max(np.abs(shell), axis=0)
    radius = np.linalg.norm(shell, axis=1).max()
    cache = UsdGeom.XformCache()
    collisions = {}
    for tick in range(0, round(stage.GetEndTimeCode() * 2) + 1, stride):
        frame = tick * 0.5
        cache.SetTime(frame)
        mats = np.stack(
            [np.asarray(cache.GetLocalToWorldTransform(piece)) for piece in pieces]
        )
        centres = mats[:, 3, :3]
        for hand, (prim, verts, faces, hull) in enumerate(fingers):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            boundary = hull @ matrix[:3, :3] + matrix[3, :3]
            nearest = (
                np.clip(centres, boundary.min(axis=0), boundary.max(axis=0)) - centres
            )
            near = np.einsum("ij,ij->i", nearest, nearest) < radius**2
            for j, begin, end in legal[hand // 2]:
                if begin <= frame <= end:
                    near[j] = False
            for j in np.flatnonzero(near):
                world = verts @ matrix[:3, :3] + matrix[3, :3]
                local = (world - centres[j]) @ mats[j, :3, :3].T
                if intersects(local, faces, planes, extent):
                    collisions.setdefault((hand, j), []).append(frame)
    for (hand, j), frames in collisions.items():
        print(
            f"Finger {hand} touched non-target {pieces[j].GetName()} at scene frames "
            f"{min(frames):g}..{max(frames):g} ({len(frames)} samples)",
            flush=True,
        )
    if fail and collisions:
        raise AssertionError("Finger swept through a non-target token")
    print(
        f"Neighbour check: {count} pieces, 4 fingers, {len(range(0, round(stage.GetEndTimeCode() * 2) + 1, stride))} samples, "
        f"{len(collisions)} interfering pairs.",
        flush=True,
    )
    return collisions
