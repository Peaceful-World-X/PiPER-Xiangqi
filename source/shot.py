"""Shared playback and sampling clock, in seconds."""

FPS = 50
VIDEO_FPS = 30
REFERENCE_COUNT = 4013
FRAME_COUNT = 4800
LAST = FRAME_COUNT - 1
DURATION = FRAME_COUNT / FPS


def camera_samples(track, reference_time=None):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Reconstruct the hand-held camera on a uniform clock. Filtering the pose
    # derivative removes the speed change at each 5 Hz tracking observation.
    t = np.arange(REFERENCE_COUNT) / FPS
    result = []
    for path in (track["eyes"], track["rot"]):
        dense = np.column_stack([np.interp(t, track["time"], x) for x in path.T])
        pad = 20
        head = dense[0] - np.arange(pad, 0, -1)[:, None] * (dense[1] - dense[0])
        tail = dense[-1] + np.arange(1, pad + 1)[:, None] * (dense[-1] - dense[-2])
        result.append(
            gaussian_filter1d(np.vstack([head, dense, tail]), 5.5, axis=0)[pad:-pad]
        )
    if reference_time is not None:
        result = [np.column_stack([np.interp(reference_time,t,channel) for channel in a.T]) for a in result]
    return result
