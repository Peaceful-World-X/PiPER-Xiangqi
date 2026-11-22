"""Launch an editable Isaac Sim viewport on the animated shot camera."""

import argparse
from pathlib import Path
from shot import DURATION, FPS, VIDEO_FPS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--verify", action="store_true", help="check playback headlessly, then exit")
parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parents[1] / "Workshop.usdc")
args, _ = parser.parse_known_args()
verify = args.verify
stage_path = args.stage.resolve()
if not stage_path.is_file():
    parser.error(f"{stage_path} is missing. Run source/build_workshop.py with the project .venv first.")

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": verify,
        "hide_ui": False,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
        "multi_gpu": False,
    }
)
import carb, omni.usd, omni.timeline
from omni.kit.viewport.utility import get_active_viewport

ctx = omni.usd.get_context()
if not ctx.open_stage(str(stage_path)):
    raise RuntimeError(f"Could not open {stage_path}")
for _ in range(10):
    app.update()
s = carb.settings.get_settings()
s.set("/rtx/post/tonemap/op", 4)
s.set("/rtx/post/tonemap/filmIso", 600.0)
s.set("/rtx/post/tonemap/whitepoint", 6500.0)
s.set("/app/viewport/grid/enabled", False)
s.set("/rtx/post/dlss/execMode", 2)
vp = get_active_viewport()
vp.camera_path = "/World/Camera"
vp.resolution = (1280, 720)
timeline = omni.timeline.get_timeline_interface()
s.set("/app/player/useFixedTimeStepping", False)
s.set("/app/runLoops/main/rateLimitEnabled", True)
s.set("/app/runLoops/main/rateLimitFrequency", VIDEO_FPS)
timeline.set_time_codes_per_second(FPS)
timeline.set_ticks_per_frame(1)
timeline.set_start_time(0.0)
timeline.set_end_time(DURATION)
timeline.set_target_framerate(VIDEO_FPS)
timeline.set_play_every_frame(False)
timeline.set_current_time(0.0)
timeline.set_looping(True)
timeline.play()
if verify:
    times = []
    for _ in range(20):
        app.update()
        times.append(timeline.get_current_time())
    assert timeline.get_time_codes_per_seconds() == FPS
    steps = [b - a for a, b in zip(times, times[1:])]
    assert min(steps) >= 0 and times[-1] > times[0], steps
    assert str(vp.camera_path) == "/World/Camera"
    assert (
        ctx.get_stage()
        .GetPrimAtPath(
            "/World/RightPiPER/base_link/link1/link2/link3/link4/link5/link6"
        )
        .IsValid()
    )
    print(
        "VERIFIED playback",
        timeline.is_playing(),
        timeline.get_current_time(),
        "seconds, time codes per second",
        timeline.get_time_codes_per_seconds(),
        flush=True,
    )
else:
    while app.is_running():
        app.update()
app.close()
