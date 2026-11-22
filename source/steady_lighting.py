"""Reproject low-frequency lighting between consecutive RGB renders."""
from collections import deque

import cv2 as cv
import numpy as np


class LightingHistory:
    def __init__(self, width, height):
        self.size = ((width + 1) // 2, (height + 1) // 2)
        self.flow_size = ((width + 3) // 4, (height + 3) // 4)
        self.scale = np.float32(self.size) / self.flow_size
        self.grid = self.pixel_grid(self.size)
        self.flow_grid = self.pixel_grid(self.flow_size)
        self.solver = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_FAST)
        self.solver.setFinestScale(0)

    @staticmethod
    def pixel_grid(size):
        y, x = np.indices(size[::-1], dtype=np.float32)
        return np.dstack((x, y))

    def frame(self, image):
        # Some render products reuse the readback buffer on the next frame.
        image = image.copy()
        low = cv.GaussianBlur(cv.pyrDown(image), (3, 3), .65)
        grey = cv.resize(cv.cvtColor(low, cv.COLOR_RGB2GRAY), self.flow_size,
                         interpolation=cv.INTER_AREA)
        full_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        peaks = cv.dilate(full_grey, np.ones((3,3), np.uint8))
        return {"image": image, "low": low, "grey": grey, "peaks": peaks}

    def mapping(self, flow, reverse):
        inverse = cv.remap(reverse, self.flow_grid + flow, None, cv.INTER_LINEAR,
                           borderMode=cv.BORDER_REPLICATE)
        error = np.minimum(np.square(flow + inverse).sum(axis=2), 8)
        confidence = cv.resize(np.exp(-1.6 * error), self.size)
        coords = self.grid + cv.resize(flow, self.size) * self.scale.astype(np.float32)
        x, y = coords[:, :, 0], coords[:, :, 1]
        confidence[(x < 0) | (y < 0) | (x > self.size[0]-1) | (y > self.size[1]-1)] = 0
        return cv.convertMaps(coords, None, cv.CV_16SC2), confidence

    def link(self, previous, current):
        backward = self.solver.calc(current["grey"], previous["grey"], None)
        forward = self.solver.calc(previous["grey"], current["grey"], None)
        for dest, src, flow, reverse, key in (
            (current, previous, backward, forward, "past"),
            (previous, current, forward, backward, "future"),
        ):
            maps, confidence = self.mapping(flow, reverse)
            warped = cv.remap(src["low"], *maps, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
            difference = cv.absdiff(dest["low"], warped).max(axis=2).astype(np.float32)
            # Shadow boundaries and newly revealed surfaces should use the current frame.
            local = cv.blur(difference, (5, 5))
            confidence *= np.exp(-.5 * ((local / 9)**2 + (difference / 13)**2))
            dest[key] = maps, confidence
            dest[key + "_peaks"] = src["peaks"]

    def merge(self, current, prior, direction):
        base = current["low"].astype(np.float32)
        if prior is None:
            return base, np.ones(base.shape[:2], np.float32)
        maps, confidence = current[direction]
        history, mass = prior
        mass = cv.remap(mass, *maps, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        mass = np.minimum(mass * confidence * .84, 6)
        history = cv.remap(history, *maps, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        delta = np.clip(history - base, -15, 15)
        return base + delta * (mass / (1 + mass))[:, :, None], 1 + mass

    def remove_sparks(self, sample):
        image = sample["image"]
        if "past_peaks" not in sample or "future_peaks" not in sample:
            return image
        grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        median = cv.medianBlur(grey, 3)
        y, x = np.where(cv.subtract(grey, median) > 15)
        if not len(y):
            return image
        # Optical flow is computed at half size. Look up the full-size local
        # maximum on BOTH sides so a printed stroke or moving highlight survives.
        stable = np.ones(len(y),dtype=bool)
        height,width = grey.shape
        for key in ("past","future"):
            (integer, fraction), confidence = sample[key]
            hy,hx = y//2,x//2
            frac = fraction[hy,hx]
            origin = integer[hy,hx].astype(np.float32)
            origin[:,0] += (frac%32)/32
            origin[:,1] += (frac//32)/32
            sx = np.rint(x+(origin[:,0]-hx)*2).astype(int)
            sy = np.rint(y+(origin[:,1]-hy)*2).astype(int)
            valid=(sx>=1)&(sx<width-1)&(sy>=1)&(sy<height-1)
            reference = sample[key+"_peaks"][sy.clip(0,height-1),sx.clip(0,width-1)]
            stable &= valid & (confidence[hy,hx]>.28) & (grey[y,x].astype(int)>reference.astype(int)+11)
        if stable.any():
            y,x=y[stable],x[stable]
            image[y,x] = cv.medianBlur(image,3)[y,x]
        return image

    def output(self, window, past):
        future = None
        for sample in reversed(window):
            future = self.merge(sample, future, "future")
        current = window[0]
        ahead, ahead_weight = future
        behind, behind_weight = past
        low = current["low"].astype(np.float32)
        delta = (ahead - low) * ahead_weight[:, :, None] + (behind - low) * behind_weight[:, :, None]
        delta /= (ahead_weight + behind_weight - 1)[:, :, None]
        image = self.remove_sparks(current)
        delta = cv.resize(delta, (image.shape[1], image.shape[0]))
        return np.clip(np.rint(image.astype(np.float32) + delta), 0, 255).astype(np.uint8)


def steady_lighting(frames):
    """Stabilize broad shading while retaining the current frame's fine detail."""
    window = deque()
    model = None
    past = None
    for image in frames:
        if min(image.shape[:2]) < 48:
            yield image
            continue
        if model is None:
            model = LightingHistory(image.shape[1], image.shape[0])
        sample = model.frame(image)
        if window:
            model.link(window[-1], sample)
        window.append(sample)
        if len(window) == 5:
            past = model.merge(window[0], past, "past")
            yield model.output(window, past)
            window.popleft()
    while window:
        past = model.merge(window[0], past, "past")
        yield model.output(window, past)
        window.popleft()
