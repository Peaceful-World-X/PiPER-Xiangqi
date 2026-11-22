"""Approximate a short, centred exposure using neighbouring rendered RGB frames."""
import cv2 as cv
import numpy as np


def expose(frames, fps):
    if fps > 35:
        yield from frames
        return
    frames = iter(frames)
    centre = next(frames, None)
    if centre is None:
        return
    height, width = centre.shape[:2]
    small = (max(32, width // 3), max(24, height // 3))
    gy, gx = np.indices((height, width), dtype=np.float32)
    yy, xx = np.indices(small[::-1], dtype=np.float32)
    flow = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_FAST)
    cv.setNumThreads(3)

    def luminance(image):
        return cv.cvtColor(cv.resize(image, small, interpolation=cv.INTER_AREA), cv.COLOR_RGB2GRAY)

    def velocity(field, reverse):
        inverse = cv.remap(reverse, xx + field[:, :, 0], yy + field[:, :, 1],
                           cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        # Suppress mismatched flow where a moving finger reveals the board.
        error = np.linalg.norm(field + inverse, axis=2)
        confidence = np.exp(-0.50 * error ** 2)
        field = cv.resize(field * confidence[:, :, None], (width, height))
        field *= np.array([width / small[0], height / small[1]], np.float32)
        return np.clip(field, -64, 64)

    def sample(image, field):
        return cv.remap(image, gx + .17 * field[:, :, 0], gy + .17 * field[:, :, 1],
                        cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    grey = luminance(centre)
    previous = None
    for following in frames:
        next_grey = luminance(following)
        forward = flow.calc(grey, next_grey, None)
        backward = flow.calc(next_grey, grey, None)
        ahead = velocity(forward, backward)
        behind = previous if previous is not None else -ahead
        sides = cv.addWeighted(sample(centre, ahead), .5, sample(centre, behind), .5, 0)
        yield cv.addWeighted(centre, .44, sides, .56, 0)
        previous = velocity(backward, forward)
        centre, grey = following, next_grey
    yield centre
