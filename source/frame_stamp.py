"""Identify each native RTX readback using marks outside the exported camera gate."""

from collections import deque

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade


class FrameStamp:
    padding = 20

    def __init__(self, stage, width, height, first, count, scene_step):
        self.width = width
        self.height = height
        self.first = first
        self.count = count
        self.scene_step = scene_step
        self.bits = max(1, count.bit_length())
        strip_width = max(min(width, 140), 25 + (self.bits + 1) * 9)
        self.origin = (width - strip_width) // 2
        if self.origin < 0:
            raise ValueError("Frame stamp does not fit within the overscan strip")
        camera = UsdGeom.Camera.Get(stage, "/World/Camera")
        focal_pixels = (
            width
            * camera.GetFocalLengthAttr().Get()
            / camera.GetHorizontalApertureAttr().Get()
        )
        self.bottom = height + 2 * self.padding
        asset_root = "/World/CaptureStampMaterials"
        stamp_root = "/World/Camera/CaptureStamp"

        def material(name, intensity):
            root = asset_root + "/" + name
            mat = UsdShade.Material.Define(stage, root)
            shader = UsdShade.Shader.Define(stage, root + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                (0, 0, 0)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                (intensity,) * 3
            )
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            return mat

        def card(name, x, y, dx, dy, depth, mat):
            mesh = UsdGeom.Mesh.Define(stage, stamp_root + "/" + name)
            mesh.CreatePointsAttr(
                [
                    (
                        (u - width / 2) * depth / focal_pixels,
                        (self.bottom / 2 - v) * depth / focal_pixels,
                        -depth,
                    )
                    for u, v in [(x, y + dy), (x + dx, y + dy), (x + dx, y), (x, y)]
                ]
            )
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            return mesh

        rate = stage.GetTimeCodesPerSecond()
        camera.GetVerticalApertureAttr().Set(
            camera.GetHorizontalApertureAttr().Get() * self.bottom / width
        )
        white, black = material("White", 3), material("Black", 0)
        start = self.origin
        card(
            "Backing",
            start,
            self.bottom - 19,
            strip_width,
            18,
            0.024,
            black,
        )
        card("Reference", start + 6, self.bottom - 16, 6, 10, 0.023, white)
        for bit in range(self.bits + 1):
            pane = card(
                "Bit" + str(bit),
                start + 20 + bit * 9,
                self.bottom - 16,
                7,
                10,
                0.023,
                white,
            )
            move = pane.AddTranslateOp()
            for j in range(count):
                code = j + 1
                active = (code >> bit) & 1 if bit < self.bits else code.bit_count() % 2
                move.Set(
                    Gf.Vec3d(0, 0 if active else -0.004, 0),
                    (first + j) * scene_step * rate,
                )

    def identify(self, pixels):
        if pixels.shape != (self.bottom, self.width, 4):
            return None
        sample = pixels[-14:-8, :, 1]
        start = self.origin
        white = sample[:, start + 8 : start + 11].mean()
        black = sample[:, start + 15 : start + 18].mean()
        if white - black < 100 or black > 60:
            return None
        levels = np.array(
            [
                sample[:, start + 22 + bit * 9 : start + 25 + bit * 9].mean()
                for bit in range(self.bits + 1)
            ]
        )
        ratios = (levels - black) / (white - black)
        if np.min(np.abs(ratios - 0.5)) < 0.20:
            return None
        values = (ratios > 0.5).astype(int)
        code = int(values[:-1] @ (1 << np.arange(self.bits)))
        if not 1 <= code <= self.count or values[-1] != code.bit_count() % 2:
            return None
        return code - 1

    def samples(self, tick, check_workshop=False, missing=None):
        """Yield (global frame ID, RGB view) in actual readback order."""
        cached = missing is not None
        remaining = ({index - self.first for index in missing} if cached
                     else set(range(self.count)))

        def capture(j, recent, trace=False):
            pixels = tick((self.first + j) * self.scene_step)
            k = self.identify(pixels)
            wanted = k in remaining and k in recent
            if not wanted and not trace:
                return None
            exposure = "unavailable"
            valid = True
            if pixels.shape == (self.bottom, self.width, 4):
                frame = pixels[self.padding : -self.padding, :, :3]
                if check_workshop or trace:
                    sample = frame[::12, ::12]
                    mean = sample.mean()
                    white = (sample.min(axis=2) > 240).mean()
                    valid = not check_workshop or (65 < mean < 164 and white <= .16)
                    exposure = f"mean={mean:.2f} white={white:.3%}"
            if trace:
                code = None if k is None else k + 1
                index = None if k is None else self.first + k
                print(f"FIXED target={self.first+j} barcode={code} frame={index} "
                      f"shape={pixels.shape} {exposure} accepted={wanted and valid}", flush=True)
            if wanted and valid:
                remaining.remove(k)
                return self.first + k, frame
            return None

        def fixed(j):
            for attempt in range(21):
                item = capture(j, () if attempt < 4 else (j,), trace=cached)
                if item is not None:
                    return item
            raise RuntimeError(f"RTX failed to read back frame {self.first + j}")

        if cached and remaining:
            # Settle at the first missing frame before advancing the clock.
            yield fixed(min(remaining))
        for first in range(0, self.count, 72):
            ids = range(first, min(first + 72, self.count))
            if not remaining.intersection(ids):
                continue
            if cached:
                print(f"BLOCK {self.first+first}..{self.first+ids.stop-1} "
                      f"ready={self.count-len(remaining)}/{self.count}", flush=True)
            for ticks in (2, 3):
                pending = [j for j in ids if j in remaining]
                if not pending or (ticks == 3 and len(pending) < 4):
                    break
                recent = deque(maxlen=11)
                for n, j in enumerate([*pending, *([pending[-1]] * 4)], 1):
                    for _ in range(ticks):
                        recent.append(j)
                        item = capture(j, recent)
                        if item is not None:
                            yield item
                    if cached and n % 12 == 0:
                        print(f"CAPTURE target={self.first+j} pass={ticks} "
                              f"ready={self.count-len(remaining)}/{self.count}", flush=True)
            for j in ids:
                if j in remaining:
                    yield fixed(j)
            if cached:
                print(f"BLOCK complete ready={self.count-len(remaining)}/{self.count}", flush=True)

    def frames(self, tick, check_workshop=False):
        ready = {}
        next_frame = self.first
        for index, frame in self.samples(tick, check_workshop):
            ready[index] = frame.copy()
            while next_frame in ready:
                yield ready.pop(next_frame)
                next_frame += 1
