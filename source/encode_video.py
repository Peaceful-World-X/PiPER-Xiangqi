"""Encode ordered RGB frames with the v4 temporal filters and H.264 settings."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from shutter import expose
from steady_lighting import steady_lighting


def encode(frames, destination, width, height, fps, count, stabilize, ffmpeg=None):
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if stabilize:
        frames = steady_lighting(frames)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        partial = Path(directory) / "video.mp4"
        command = [
            ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-map", "0:v", "-an",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264",
            "-preset", "fast", "-crf", "16", "-threads", "5", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
            "-movflags", "+faststart", str(partial),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            written = 0
            for image in expose(frames, fps):
                if image.shape != (height, width, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB frame {written}")
                encoder.stdin.write(image.tobytes())
                written += 1
            if written != count:
                raise ValueError(f"Expected {count} frames, received {written}")
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg failed to encode the movie")
        partial.replace(destination)


def read_rgb(width, height):
    size = width * height * 3
    while data := sys.stdin.buffer.read(size):
        if len(data) != size:
            raise EOFError("Truncated RGB frame from Isaac Sim")
        yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    for name in ("width", "height", "fps", "count"):
        parser.add_argument(name, type=int)
    parser.add_argument("--stabilize", action="store_true")
    args = parser.parse_args()
    encode(read_rgb(args.width, args.height), args.destination, args.width, args.height,
           args.fps, args.count, args.stabilize)
