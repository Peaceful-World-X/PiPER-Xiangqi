"""Render native USD time samples through Isaac Sim RTX and encode an MP4."""

from shot import DURATION, VIDEO_FPS
import argparse
import math
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1080)
p.add_argument("--fps", type=int, default=VIDEO_FPS, help="output frame rate (default: %(default)s)")
p.add_argument("--speed", type=float, default=1.5, help="scene seconds per exported second (default: %(default)s)")
p.add_argument("--start", type=float, default=0.0, help="first shot time, in seconds")
p.add_argument("--end", type=float, default=DURATION, help="exclusive end time, in seconds")
p.add_argument("--stills", action="store_true", help="save reference views in renders/")
p.add_argument("--still-times", type=float, nargs="+", default=[0, 0.28, 3.1, 12, 36, 60, 80.2], help="reference times in seconds for --stills")
p.add_argument("--stage", default=str(root / "Workshop.usdc"), help="input USD stage")
p.add_argument("--movie", default=str(root / "renders" / "PiPER-Xiangqi.mp4"), help="output MP4 path")
p.add_argument("--path-tracing", action="store_true", help="use RTX path tracing")
p.add_argument("--frame-cache", type=Path, nargs="?", const=root / "renders" / ".frame-cache",
               metavar="DIR", help="resume verified PNG capture (default directory: renders/.frame-cache)")
args = p.parse_args()
if args.fps < 1 or not 0 <= args.start < args.end <= DURATION:
    p.error("Select a positive frame rate and an interval inside the shot.")
if not math.isfinite(args.speed) or args.speed <= 0:
    p.error("Playback speed must be positive and finite.")
scene_step = args.speed / args.fps
first_frame = round(args.start / scene_step)
count = round(args.end / scene_step) - first_frame
if count < 1:
    p.error("The selected interval contains no output frames.")
if args.width < 1 or args.height < 1:
    p.error("Width and height must be positive.")
if not args.stills and (args.width % 2 or args.height % 2):
    p.error("H.264 video requires an even width and height.")
if not Path(args.stage).is_file():
    p.error(f"USD stage not found: {args.stage}. Build Workshop.usdc with source/build_workshop.py.")


project_python = root / ".venv" / "Scripts" / "python.exe"
if args.stills:
    (root / "renders").mkdir(exist_ok=True)
elif not project_python.is_file():
    # Standalone Isaac installs can opt into their own OpenCV and FFmpeg.
    fallback_ffmpeg = shutil.which(os.getenv("IMAGEIO_FFMPEG_EXE") or "ffmpeg")
    if not fallback_ffmpeg:
        p.error("Install requirements.txt in .venv or provide IMAGEIO_FFMPEG_EXE.")


verified_movie = args.path_tracing and not args.stills and args.width >= 25 + (count.bit_length()+1)*9
if args.frame_cache is not None and not verified_movie:
    p.error("--frame-cache requires a path-traced movie wide enough for the verification stamp.")
capture_height = args.height + 40 if verified_movie else args.height

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "hide_ui": False,
        "renderer": "PathTracing" if args.path_tracing else "RayTracedLighting",
        "samples_per_pixel_per_frame": 12,
        "width": args.width,
        "height": capture_height,
        "multi_gpu": False,
        "sync_loads": True,
        "extra_args": [
            "--/app/asyncRendering=false",
            "--/app/hydraEngine/waitIdle=true",
            "--/app/updateOrder/checkForHydraRenderComplete=1000",
            "--/app/runLoops/main/rateLimitEnabled=false",
            "--/app/player/useFixedTimeStepping=false",
            "--/rtx-transient/dlssg/enabled=false",
        ],
    }
)
import carb, omni.usd, omni.timeline
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport
import numpy as np

settings = carb.settings.get_settings()
ctx = omni.usd.get_context()
if verified_movie:
    # Finish authoring the overlay before Kit loads its animated mesh topology.
    # Keep the source stage untouched and match the output clock exactly.
    import tempfile
    from pxr import Sdf, Usd
    from frame_stamp import FrameStamp

    scratch = tempfile.TemporaryDirectory(prefix="piper_capture_", ignore_cleanup_errors=True)
    overlay = Sdf.Layer.CreateNew(str(Path(scratch.name) / "capture.usda"))
    overlay.subLayerPaths = [Path(args.stage).resolve().as_posix()]
    composition = Usd.Stage.Open(overlay)
    composition.SetTimeCodesPerSecond(1/scene_step)
    composition.SetFramesPerSecond(1/scene_step)
    composition.SetStartTimeCode(0)
    composition.SetEndTimeCode(first_frame+count)
    stamp = FrameStamp(composition, args.width, args.height, first_frame, count, scene_step)
    overlay.Save()
    assert ctx.open_stage(overlay.identifier)
    del composition, overlay
else:
    assert ctx.open_stage(args.stage)
stage = ctx.get_stage()
settings.set("/rtx/post/tonemap/op", 4)
settings.set("/rtx/post/tonemap/filmIso", 600.0)
settings.set("/app/viewport/grid/enabled", False)
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set("/rtx/post/dlss/execMode", 2)
settings.set("/rtx/post/tonemap/whitepoint", 6500.0)
settings.set("/rtx/pathtracing/spp", 12)
settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
settings.set("/rtx/pathtracing/totalSpp", 64 if args.stills else 12)
settings.set("/rtx/pathtracing/fireflyFilter/enabled", True)
settings.set("/rtx/pathtracing/fireflyFilter/maxUnexposedIntensityPerSample", 8.0)
settings.set("/rtx/pathtracing/fireflyFilter/maxUnexposedIntensityPerSampleDiffuse", 6.0)
settings.set("/rtx/pathtracing/maxBounces", 4)
settings.set("/rtx/pathtracing/optixDenoiser/temporalMode/enabled", False)
settings.set("/rtx/post/motionblur/enabled", False)
viewport = get_active_viewport()
assert viewport is not None
viewport.camera_path = "/World/Camera"
viewport.resolution = (args.width, capture_height)
viewport.updates_enabled = True
timeline = omni.timeline.get_timeline_interface()
timeline.pause()
if not verified_movie:
    for _ in range(30):
        app.update()
product = None
rgb = None


def prepare_annotator():
    global product, rgb
    rep.orchestrator.set_capture_on_play(False)
    product = rep.create.render_product(
        "/World/Camera", (args.width, args.height), name="Shot"
    )
    rgb = rep.annotators.get("rgb")
    rgb.attach(product)
    viewport.updates_enabled = False
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, wait_for_render=True)
        if np.asarray(rgb.get_data()).size:
            return
    raise RuntimeError("RGB annotator did not receive a frame")


def get(t):
    timeline.set_current_time(t)
    timeline.commit()
    rep.orchestrator.step(rt_subframes=2, delta_time=0.0, wait_for_render=True)
    pixels = np.asarray(rgb.get_data())
    assert pixels.shape == (args.height, args.width, 4), pixels.shape
    return pixels[:, :, :3]


@contextmanager
def stamped_reader():
    rep.orchestrator.set_capture_on_play(False)
    annotator = rep.annotators.get("rgb")
    annotator.attach([viewport.render_product_path])

    def tick(t):
        timeline.set_current_time(t)
        timeline.commit()
        rep.orchestrator.step(rt_subframes=1,delta_time=0.,wait_for_render=False)
        return np.asarray(annotator.get_data())

    try:
        if args.frame_cache is None:
            for _ in range(12):
                tick(first_frame*scene_step)
        check_workshop = bool(stage.GetPrimAtPath("/World/Camera").GetAttribute("shot:referenceSeconds"))
        yield tick, check_workshop
    finally:
        annotator.detach([viewport.render_product_path])


def render_frames(count):
    if verified_movie:
        with stamped_reader() as (tick, check_workshop):
            yield from stamp.frames(tick, check_workshop)
        return
    # Use a dedicated render product for unstamped captures.
    prepare_annotator()
    for i in range(count):
        yield get((first_frame + i) * scene_step)



if args.stills:
    from PIL import Image, ImageFilter

    prepare_annotator()
    for t in args.still_times:
        source = stage.GetPrimAtPath("/World/Camera").GetAttribute("shot:referenceSeconds")
        keys = source.GetTimeSamples() if source else []
        screen = float(np.interp(t, [source.Get(k) for k in keys], keys)) / stage.GetTimeCodesPerSecond() if keys else t
        img = get(screen)
        Image.fromarray(img).filter(ImageFilter.GaussianBlur(0.45)).save(root / "renders" / f"frame_{t:g}.jpg", quality=95)
        print("CHECK", t, img.mean(), flush=True)
else:
    if args.frame_cache is None:
        frames = render_frames(count)
    else:
        from frame_cache import FrameCache, signature
        import omni.kit.app

        key = signature(args.stage, {
            "width": args.width, "height": args.height, "fps": args.fps, "speed": args.speed,
            "first": first_frame, "count": count, "scene_step": scene_step,
            "kit": omni.kit.app.get_app().get_build_version(),
        })
        cache = FrameCache(args.frame_cache, key, args.width, args.height, first_frame, count)
        missing = cache.missing()
        if missing:
            with stamped_reader() as (tick, check_workshop):
                for index, frame in stamp.samples(tick, check_workshop, missing):
                    cache.save(index, frame)
            if cache.missing():
                raise RuntimeError("Capture cache incomplete")
        print(f"ENCODE {count} verified PNGs", flush=True)
        frames = cache.frames()

    def captured_frames():
        for i, image in enumerate(frames):
            if i % 120 == 0:
                print("FRAME", i, "/", count, flush=True)
            yield image

    if project_python.is_file():
        command = [str(project_python), "-E", "-s", str(root / "source" / "encode_video.py"),
                   args.movie, str(args.width), str(args.height), str(args.fps), str(count)]
        if args.path_tracing:
            command.append("--stabilize")
        with subprocess.Popen(command, stdin=subprocess.PIPE) as worker:
            for frame in captured_frames():
                worker.stdin.write(frame.tobytes())
            worker.stdin.close()
            if worker.wait():
                raise RuntimeError("The project video encoder failed")
    else:
        from encode_video import encode
        encode(captured_frames(), args.movie, args.width, args.height, args.fps,
               count, args.path_tracing, fallback_ffmpeg)
print("RENDER COMPLETE", flush=True)
if product is not None:
    rgb.detach(product)
    product.destroy()
if verified_movie:
    ctx.close_stage()
    scratch.cleanup()
app.close()
