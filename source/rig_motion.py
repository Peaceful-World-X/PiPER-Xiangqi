"""Construct continuous joint-space transfers with fixed grasp and handover poses."""

from dataclasses import dataclass
import mujoco as mj
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize, linprog
from scipy.spatial.transform import Rotation

BASE_ROTATION = Rotation.from_euler("z", 90, degrees=True)
TOOL_LENGTH = 0.1368
PIECE_RADIUS = 0.0182
PIECE_HEIGHT = 0.0108
CLOSED_GAP = PIECE_RADIUS - 0.00005
from shot import FPS, FRAME_COUNT, LAST, REFERENCE_COUNT
from exchange_fit import fit_exchange
from plan_clear import plan_approach
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

END = (REFERENCE_COUNT - 1) / FPS
DWELL = 0.10
# Shot choreography limits. These govern animation pacing, not hardware controls.
VELOCITY = np.deg2rad([180, 170, 180, 210, 205, 230])
ACCELERATION = np.deg2rad([950, 950, 1050, 1350, 1250, 1550])


@dataclass
class Contact:
    key: tuple
    time: float
    xyz: np.ndarray
    axis: np.ndarray
    q: np.ndarray
    lift_q: np.ndarray


class Arm:
    def __init__(self, model, base, home):
        self.model = model
        self.data = mj.MjData(model)
        self.base = base
        self.home = home.copy()
        self.home[3:] = [0, home[4], 0]
        self.home[0] = 0
        self.limits = model.jnt_range[:6].T.copy()
        self.limits[0] += 0.00001
        self.limits[1] -= 0.00001

    def fk(self, q):
        self.data.qpos[:6] = q
        mj.mj_kinematics(self.model, self.data)
        w = self.data.body("link6")
        axes = BASE_ROTATION.as_matrix() @ w.xmat.reshape(3, 3)
        return self.base + BASE_ROTATION.apply(w.xpos) + TOOL_LENGTH * axes[:, 2], axes

    def elbow(self, eye=None):
        joint = self.data.body("link3")
        centre = self.base + BASE_ROTATION.apply(joint.xpos)
        if eye is None:
            return centre
        axle = BASE_ROTATION.apply(joint.xmat.reshape(3, 3)[:, 2])
        return centre + np.sign(np.dot(eye - centre, axle)) * .04748 * axle

    def solve(self, xyz, axis, seed=None, jaw=None):
        delta = BASE_ROTATION.inv().apply(xyz - self.base)
        yaw = np.arctan2(delta[1], delta[0])
        nominal = np.array([yaw, 1.60, -0.82, 0, 0.78, 0.0])
        if abs(axis[2]) < 0.4:
            nominal[2:5] = [-0.72, 0.0, -0.66]
        start = nominal if seed is None else seed

        def residual(q):
            tip, axes = self.fk(q)
            return np.r_[
                600 * (tip - xyz), 7 * (axes[:, 2] - axis), 0.007 * (q - nominal)
            ]

        q = least_squares(
            residual,
            np.clip(start, *self.limits),
            bounds=tuple(self.limits),
            max_nfev=100,
        ).x
        if jaw is not None:
            _, axes = self.fk(q)
            d = axes.T @ jaw
            theta = np.arctan2(-d[0], d[1])
            q[5] += ((theta + np.pi / 2) % np.pi) - np.pi / 2
        return q

    def contact(self, key, time, xyz, axis, jaw=None, seed=None, clearance=.045):
        xyz = np.asarray(xyz, float)
        q = self.solve(xyz, axis, seed=seed, jaw=jaw)
        if key[1] == "pass":
            waypoint = xyz - axis * clearance + [0.0, 0.0, 0.019]
        else:
            waypoint = xyz + [0.0, 0.0, 0.050]
        lift = self.solve(waypoint, axis, q, jaw)
        return Contact(key, time, xyz, axis, q, lift)

    def rest(self, key, time, q=None):
        q = self.home if q is None else q
        tip, axes = self.fk(q)
        return Contact((key, "rest"), time, tip, axes[:, 2], q, q)

    def filmed_rest(self, camera, motion, index, side):
        basis = Rotation.from_rotvec(camera["rot"][index]).as_matrix() @ np.diag([1, -1, -1])
        observed = motion["observed"][index, 2*side:2*side+2] * 3
        def residual(q):
            tip, axes = self.fk(q)
            elbow = self.elbow()
            view = (np.array([elbow, tip]) - camera["eyes"][index]) @ basis
            pixels = view[:, :2] / view[:, 2:] * camera["K"][0, 0] + camera["K"][:2, 2]
            return np.r_[(pixels - observed).ravel(), 18*(axes[:, 2] - [0,0,-1]),
                         4*(q - self.home)]
        return least_squares(residual, np.clip(self.home, *self.limits), bounds=tuple(self.limits)).x


def transition(first, last, arm=None):
    values = [first.q]
    if first.key[1] != "rest":
        values.append(first.lift_q)
    if last.key[1] != "rest":
        values.append(last.lift_q)
    values.append(last.q)
    distances = np.linalg.norm(np.diff(values, axis=0) / VELOCITY, axis=1)
    knots = np.r_[0.0, np.cumsum(np.maximum(distances, 0.012) ** 0.70)]
    knots /= knots[-1]
    # Monotone interpolation stays inside the manufacturer's joint limits.
    slopes = PchipInterpolator(knots, values, axis=0).derivative()(knots)
    slopes[[0, -1]] = 0.0
    spline = CubicHermiteSpline(knots, values, slopes, axis=0)
    # Parameterise by joint speed and acceleration. A single difficult turn
    # should slow only the affected part of a transfer.
    stations = np.unique(np.round(np.r_[np.linspace(0, 1, 61), knots], 10))
    du = np.diff(stations)
    n = len(du)
    dq = spline(stations, 1)
    vmax = np.minimum(4.0, np.min(VELOCITY / (np.abs(dq) + 1e-10), axis=1))
    if arm is not None:
        # Bound the tool speed along the CAD joint spline. A lightly rotating
        # joint can still move the tips a long way near full extension.
        points = np.array([arm.fk(q)[0] for q in spline(stations)])
        derivative = np.linalg.norm(np.gradient(points, stations, axis=0), axis=1)
        vmax = np.minimum(vmax, .58 / np.maximum(derivative, .0001))
    vmax[[0, -1]] = 0.0
    terms = []
    ids = np.arange(n)
    for u in (1e-5, 0.5, 1 - 1e-5):
        at = stations[:-1] + u * du
        vel = spline(at, 1)
        acc = spline(at, 2)
        mat = np.zeros((n, 6, n + 1))
        mat[ids, :, ids] = (1 - u) * acc - vel / (2 * du[:, None])
        mat[ids, :, ids + 1] = u * acc + vel / (2 * du[:, None])
        terms.append(mat.reshape(-1, n + 1))
    mat = np.concatenate(terms)
    # x is squared path speed. Joint acceleration is linear in neighbouring x.
    lp = linprog(
        -np.ones(n + 1),
        A_ub=np.r_[mat, -mat],
        b_ub=np.tile(ACCELERATION, 6 * n),
        bounds=list(zip(np.zeros(n + 1), vmax**2)),
        method="highs",
    )
    if not lp.success:
        raise ValueError(lp.message)
    speed = np.sqrt(np.maximum(lp.x, 0))
    durations = 2 * du / (speed[1:] + speed[:-1])
    travel = np.r_[0.0, np.cumsum(durations)]
    duration = travel[-1]

    def evaluate(u):
        t = u * duration
        ix = np.clip(np.searchsorted(travel, t, side="right") - 1, 0, n - 1)
        dt = t - travel[ix]
        param = stations[ix] + dt * (
            speed[ix] + 0.5 * dt * (speed[ix + 1] - speed[ix]) / durations[ix]
        )
        return spline(np.clip(param, 0, 1))

    return evaluate, duration * 1.07


def measured_pause(clock, tips, start, end):
    """Measure the longest low-speed interval while the empty claw is clear."""
    if end <= start + .5:
        return None
    speed = np.linalg.norm(np.diff(tips, axis=0), axis=1) / np.diff(clock)
    quiet = (speed < .10) & (clock[:-1] >= start + .12) & (clock[1:] <= end - .12)
    edges = np.diff(np.r_[False, quiet, False].astype(int))
    begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(begins):
        return None
    lengths = clock[ends] - clock[begins]
    i = lengths.argmax()
    if lengths[i] < .39:
        return None
    return (clock[begins[i]] - start) / (end - start), lengths[i] / (end - start)


def plan_motion(model, motion, camera, reference=False):
    ts = motion["time"]
    # Leave 0.8 exported seconds of clear board at the end. Plan on the
    # delivered clock so activity of the other arm cannot stretch a transfer.
    plan_end = LAST / FPS - 1.20 if reference else END
    arms = [Arm(model, b, motion["qs"][k, 0]) for k, b in enumerate(motion["base"])]
    down = np.array([0.0, 0.0, -1.0])
    visits = [[], []]
    narrow = {}
    requested = {}
    for j, (donor, receiver, source, pick, meet, drop) in enumerate(motion["plan"]):
        donor, receiver, source = map(int, (donor, receiver, source))
        start = np.r_[motion["starts"][source], motion["startz"][source] + PIECE_HEIGHT / 2]
        take_jaw, narrow[j,"take"] = plan_approach(arms[donor],motion,j,"take",start,down,BASE_ROTATION)
        take = arms[donor].contact((j, "take"), pick, start, down, take_jaw)
        landing = motion["dest"][j].copy()
        landing[2] = 0.00013 + PIECE_HEIGHT / 2
        # At a reach limit the wrist can set the disc down on its bevel.
        # Raise its centre to the actual lower rim before the fingers open.
        for _ in range(3):
            place_q = arms[receiver].solve(landing, down)
            _, rotation = arms[receiver].fk(place_q)
            cosine = np.clip(-rotation[2, 2], 0, 1)
            rise = PIECE_HEIGHT / 2 * cosine + .915 * PIECE_RADIUS * np.sqrt(1 - cosine**2)
            landing[2] = .00013 + rise
        place_jaw, narrow[j,"place"] = plan_approach(arms[receiver],motion,j,"place",landing,down,BASE_ROTATION)
        place = arms[receiver].contact((j, "place"), drop, landing, down, place_jaw)
        if meet < 0:
            visits[donor].append((pick, [take, place]))
        else:
            idx = np.argmin(abs(ts - meet))
            seed = motion["tips"][:, idx].mean(0)
            marker = motion["exchange_marker"][j] if reference else j
            center, direction, jaws, poses = fit_exchange(marker, meet, camera, arms, seed, reference)
            hand = []
            for k, arm in enumerate(arms):
                axis = direction * (1 - 2*k)
                jaw = jaws[k]
                hand.append(
                    arm.contact((j, "pass"), meet, center, axis, jaw, poses[k],
                                clearance=.087 if reference and k == receiver else .045)
                )
            visits[donor].append((pick, [take, hand[donor]]))
            visits[receiver].append((meet, [hand[receiver], place]))
        requested[(j, "take")] = pick
        requested[(j, "place")] = drop
        if meet >= 0:
            requested[(j, "pass")] = meet
    requested[("start", "rest")] = 0.0
    requested[("end", "rest")] = END
    keys = list(requested)
    ids = {key: i for i, key in enumerate(keys)}
    first_exchange = min((key for key in keys if key[1] == "pass"),
                         key=requested.get, default=None)

    def dwell(contact, arm_index):
        if contact.key[1] != "pass":
            return DWELL / 2, DWELL / 2
        donor = int(motion["plan"][contact.key[0], 0])
        # Donor arrives first. The receiver stays put until the donor retreats.
        if reference:
            return (.98, .16) if arm_index == donor else (.17, .32)
        if contact.key == first_exchange:
            return (.42, .24) if arm_index == donor else (.25, .38)
        return (.26, .17) if arm_index == donor else (.18, .24)

    tracks = []
    curves = []
    edges = []
    mins = []
    travel_floors = []
    for k, arm in enumerate(arms):
        start_q = arm.filmed_rest(camera, motion, 0, k) if reference else None
        end_q = arm.filmed_rest(camera, motion, -1, k) if reference else None
        events = [arm.rest("start", 0.0, start_q)]
        for _, pair in sorted(visits[k], key=lambda v: v[0]):
            events.extend(pair)
        events.append(arm.rest("end", END, end_q))
        paths = []
        floors = []
        for first, last in zip(events, events[1:]):
            curve, floor = transition(first, last, arm if reference else None)
            floors.append(floor)
            edges.append([ids[first.key], ids[last.key]])
            mins.append(floor + dwell(first,k)[1] + dwell(last,k)[0] + 0.025)
            paths.append(curve)
        tracks.append(events)
        curves.append(paths)
        travel_floors.append(floors)
    mat = np.zeros((len(edges), len(keys)))
    for row, (l, r) in enumerate(edges):
        mat[row, l] = -1
        mat[row, r] = 1
    ref = np.array([requested[k] for k in keys])
    if reference:
        ref *= plan_end / END
    weights = np.array(
        [3 if k[1] == "place" else 1.5 if k[1] == "pass" else 1.0 for k in keys]
    )
    lb = np.zeros(len(keys))
    ub = np.full(len(keys), plan_end)
    for key in [("start", "rest"), ("end", "rest")]:
        lb[ids[key]] = ub[ids[key]] = ref[ids[key]]
    mins = np.array(mins)
    # Both arms share the same time variable at handover. Each arm must release
    # its current piece before it can pick up another one.
    feasible = linprog(
        np.zeros_like(ref),
        A_ub=-mat,
        b_ub=-mins,
        bounds=list(zip(lb, ub)),
        method="highs",
    )
    if not feasible.success:
        raise ValueError(
            "The contact schedule needs more time for the chosen joint limits"
        )
    solve = minimize(
        lambda x: np.sum(weights * (x - ref) ** 2),
        feasible.x,
        jac=lambda x: 2 * weights * (x - ref),
        method="SLSQP",
        bounds=Bounds(lb, ub),
        constraints=[LinearConstraint(mat, mins, np.inf)],
        options={"maxiter": 400, "ftol": 1e-9},
    )
    if not solve.success:
        raise ValueError(solve.message)
    # Quantise events to output frames and keep held contact poses on both sides.
    times = {key: round(solve.x[i] * FPS) / FPS for i, key in enumerate(keys)}
    frames = np.arange(round(plan_end * FPS) + 1) if reference else np.arange(REFERENCE_COUNT)
    clock = frames / FPS
    qs = np.empty((2, len(frames), 6))
    gaps = np.empty((2, len(frames)))
    updated = motion["plan"].copy()
    for j, row in enumerate(updated):
        row[3] = times[j, "take"]
        row[5] = times[j, "place"]
        if row[4] >= 0:
            row[4] = times[j, "pass"]
    for k, arm in enumerate(arms):
        ev = tracks[k]
        paths = curves[k]
        qs[k] = arm.home
        for first, last, curve, floor in zip(ev, ev[1:], paths, travel_floors[k]):
            t0, t1 = times[first.key], times[last.key]
            mask = (clock >= t0) & (clock <= t1)
            start = t0 + dwell(first,k)[1]
            travel = t1 - dwell(last,k)[0] - start
            elapsed = np.clip(clock[mask] - start, 0, travel)
            if reference:
                # Bound the cruise time independently of work assigned to the other arm.
                points = np.array([arm.fk(q)[0] for q in curve(np.linspace(0, 1, 41))])
                length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                moving = min(travel, max(floor, length / .36 + .16))
                slack = travel - moving
                receiving = last.key[1] == "pass" and int(motion["plan"][last.key[0], 1]) == k
                empty = first.key[1] == "place" or (
                    first.key[1] == "pass" and int(motion["plan"][first.key[0], 0]) == k
                )
                if slack > .48 and empty:
                    # Retract promptly after releasing. Wait clear of both contacts,
                    # then descend just in time to close on the next token.
                    pause = measured_pause(ts, motion["tips"][k], first.time, last.time)
                    if pause is None:
                        # With no sustained pause in the tracking, use a short wait
                        # at clearance and retain motion through the transfer.
                        wait = min(slack, .32)
                        onset = .2 * travel
                    else:
                        onset, fraction = pause
                        wait = min(slack, fraction * travel)
                        onset *= travel
                    moving = travel - wait
                    brake = min(.28, .48 * wait, .12 * moving)
                    decelerate = np.clip(onset, .12 * moving, .84 * moving)
                    slack = wait
                    restart = decelerate + slack
                    down_ramp = np.clip(elapsed - decelerate, 0, brake)
                    up_ramp = np.clip(elapsed - restart, 0, brake)
                    elapsed = (np.minimum(elapsed, decelerate) + down_ramp
                               - down_ramp**2 / (2 * brake) + up_ramp**2 / (2 * brake)
                               + np.maximum(elapsed - restart - brake, 0))
                elif first.key[1] == "rest" or empty:
                    if receiving or last.key[1] == "take":
                        elapsed -= slack
                else:
                    # A held piece uses the whole transfer, up to the specified
                    # contact dwell. This avoids a sprint followed by a long hover.
                    moving = travel
                travel = moving
            u = np.clip(elapsed / travel, 0, 1)
            # Average a short window of neighbouring joint poses in time.
            # Shrink the kernel at contact so pick / handover dwells stay exact.
            window = 0.215 / travel
            span = window * np.tanh(u / window) * np.tanh((1-u)/window)
            offsets = np.linspace(-1, 1, 13)
            weights = np.exp(-0.5 * (offsets / 0.42) ** 2)
            samples = curve(u[:, None] + span[:, None] * offsets[None, :])
            qs[k, mask] = np.average(samples, axis=1, weights=weights)
        gaps[k] = 0.031
        for j, (d, r, ss, p, h, t) in enumerate(updated):
            intervals = [(int(d), p, h), (int(r), h, t)] if h >= 0 else [(int(d), p, t)]
            for owner, begin, end in intervals:
                if owner != k:
                    continue
                # Both grippers hold briefly at the handover centre.
                close = np.clip((clock - begin + .11) / .08, 0, 1)
                open_ = np.clip((end + .11 - clock) / .08, 0, 1)
                if h >= 0:
                    lead = .23 if (j, "pass") == first_exchange else .16
                    lag = .21 if (j, "pass") == first_exchange else .15
                    if reference:
                        lead, lag = .14, .12
                    if owner == int(r):
                        close = np.clip((clock - h + lead) / .10, 0, 1)
                    else:
                        open_ = np.clip((h + lag - clock) / .09, 0, 1)
                amount = np.minimum(close, open_)
                gaps[k] = np.minimum(
                    gaps[k], 0.031 - (0.031 - CLOSED_GAP) * (amount * amount * (3 - 2 * amount))
                )
            for role, owner, when in (("take", int(d), p), ("place", int(r), t)):
                if k == owner and narrow[j, role]:
                    # Keep 5.8 mm per side while descending or rising in a crowded row.
                    u = np.clip((.48 - np.abs(clock-when))/.19, 0., 1.)
                    gaps[k] = np.minimum(gaps[k], .031 - .007*u*u*(3-2*u))
    if reference:
        screen_clock = clock
    else:
        # One shared clock protects both hands and their carried piece during a handover.
        # Give high-speed turns extra screen time, with gentle changes in playback rate.
        vel = np.abs(np.gradient(qs, 1/FPS, axis=1)).max(axis=(0,2))
        acc = np.abs(np.gradient(np.gradient(qs, 1/FPS, axis=1), 1/FPS, axis=1)).max(axis=(0,2))
        load = np.maximum(vel/np.deg2rad(110), np.sqrt(acc/np.deg2rad(445)))
        demand = np.maximum(.95, maximum_filter1d(load, size=13, mode="nearest"))
        pace = gaussian_filter1d(demand, 7.0, mode="nearest")
        screen_clock = np.r_[0.,np.cumsum((pace[1:]+pace[:-1])/(2*FPS))]
        # Compress the final retreat locally, then hold the clear terminal pose for 0.8 s.
        remaining = 1.20
        tail = np.clip((clock - (END-9.0))/9.0, 0., 1.)
        screen_clock *= (LAST/FPS) / screen_clock[-1]
        screen_clock -= remaining * tail**2 * (3.-2.*tail)
        # Retain the editorial beats in the 1.5x release as paths change length.
        # Times at the launch of the exchange, close of the crowded pickup and last release.
        beats = [(10, "pass", 2.96), (5, "take", 6.24), (0, "pass", 821/75),
                 (17, "place", 2870/75)]
        before = [np.interp(times[j, role], clock, screen_clock) for j,role,_ in beats]
        last_motion_time = screen_clock[-1]
        warp = PchipInterpolator([0., *before, last_motion_time],
                                 [0., *[t*1.5 for _,_,t in beats], last_motion_time])
        screen_clock = warp(screen_clock)
    screen = np.arange(FRAME_COUNT)/FPS
    source = np.interp(screen, screen_clock, clock)
    qs = PchipInterpolator(clock,qs,axis=1)(source)
    gaps = PchipInterpolator(clock,gaps,axis=1)(source)
    for column in (3,4,5):
        used = updated[:,column]>=0
        updated[used,column] = np.rint(np.interp(updated[used,column],clock,screen_clock)*FPS)/FPS
    print("Screen duration and peak speed", round(screen[-1]+1/FPS,2),
          round(float(np.rad2deg(abs(np.diff(qs,axis=1))).max()*FPS),1),flush=True)
    # Match the board-tracked camera to the same photographed exchange as the arms.
    rows = np.flatnonzero(updated[:,4] >= 0)
    rows = rows[np.argsort(updated[rows,4])]
    anchors = np.r_[0., [times[j,"pass"] for j in rows], plan_end]
    reference_times = np.r_[0., motion["plan"][rows,4], END]
    source = PchipInterpolator(anchors, reference_times)(source)
    return qs, gaps, updated, source
