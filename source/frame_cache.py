"""Store verified camera-gate PNGs under a content-addressed capture directory."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from PIL import Image, PngImagePlugin
from pxr import UsdUtils


def signature(stage, clock_and_settings):
    stage = Path(stage).resolve()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(stage))
    if unresolved:
        raise ValueError(f"Unresolved capture dependencies: {unresolved}")
    files = {stage, *(Path(layer.realPath) for layer in layers), *map(Path, assets)}
    # The two source files define the Kit/RTX settings, overscan and acceptance rules.
    files.update(Path(__file__).with_name(name) for name in ("render_isaac.py", "frame_stamp.py"))
    contents = {}
    for path in sorted(files):
        with path.open("rb") as stream:
            contents[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = json.dumps([str(stage), clock_and_settings, contents], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class FrameCache:
    def __init__(self, root, key, width, height, first, count):
        self.directory = Path(root).resolve() / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.ids = range(first, first + count)
        print("FRAME CACHE", self.directory, flush=True)

    def path(self, index):
        return self.directory / f"{index:04d}.png"

    def save(self, index, frame):
        tag = PngImagePlugin.PngInfo()
        tag.add_text("capture", self.directory.name)
        tag.add_text("frame", str(index))
        fd, pending = mkstemp(dir=self.directory, suffix=".pending")
        try:
            with os.fdopen(fd, "wb") as stream:
                Image.fromarray(frame).save(stream, "PNG", pnginfo=tag, compress_level=1)
            os.replace(pending, self.path(index))
        finally:
            Path(pending).unlink(missing_ok=True)

    def read(self, index):
        path = self.path(index)
        with path.open("rb") as stream:
            stream.seek(-12, os.SEEK_END)
            if stream.read() != bytes.fromhex("0000000049454e44ae426082"):
                raise ValueError("missing PNG end chunk")
            stream.seek(0)
            with Image.open(stream) as image:
                if (image.format != "PNG" or image.mode != "RGB" or image.size != self.size
                        or image.info.get("capture") != self.directory.name
                        or image.info.get("frame") != str(index)):
                    raise ValueError("capture, frame ID or RGB dimensions differ")
                image.verify()  # Check chunk CRCs before accepting an existing file.
            stream.seek(0)
            with Image.open(stream) as image:
                return np.asarray(image)

    def missing(self):
        missing = set()
        for index in self.ids:
            try:
                self.read(index)
            except FileNotFoundError:
                missing.add(index)
            except (OSError, SyntaxError, ValueError) as error:
                print(f"CACHE invalid {self.path(index).name}: {error}", flush=True)
                missing.add(index)
        print(f"CACHE ready {len(self.ids) - len(missing)}/{len(self.ids)}", flush=True)
        return missing

    def frames(self):
        return (self.read(index) for index in self.ids)
