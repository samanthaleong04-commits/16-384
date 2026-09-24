"""Run the students' RRR code on the simulated xArm7, in meshcat.

Leaves joints 1, 4 and 7 with their axes all parallel to world z, so what is
left is an exact planar 3R — joint1 at the shoulder, joint4 at the elbow,
joint7 at the wrist. Those four are held by commanding them every tick, not by
welding them in the model: the arm stays a real 7-DoF arm that the safety guard
and the servos still understand, it just isn't asked to move them.

`theta` here always means the exercise's planar angles, `q` the arm's seven
joint values. `PlanarChain` is the translation between them, and it measures
itself off the loaded model rather than carrying the numbers as constants — the
link lengths, the plane heights and the per-joint sign and offset all come from
probing MuJoCo at startup, so a Menagerie model bump can't silently invalidate
them.

Nothing in here grades anything. `local_autograder.py` still decides right and
wrong from the fixtures; this only shows you what your code does.
"""

import collections
import http.server
import json
import math
import socket
import string
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# The library imports its own modules flat (`from mujoco_meshcat import ...`),
# so its directory has to be importable, not just this one.
if str(ROOT / "xarm7_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "xarm7_lib"))

import meshcat.geometry as g  # noqa: E402  (needs the sys.path above)
import mujoco  # noqa: E402

from mujoco_meshcat import MeshcatVisualizer  # noqa: E402
from xarm7_mujoco import SimulatedXArm7  # noqa: E402

# The four joints held still, by 0-based index, and what they are held at.
# Joint2's range is [-2.059, 2.0944] and joint6's is [-1.693, 3.1416]; the rest
# are +-2*pi. All four values below are comfortably inside.
LOCKED_JOINTS = {
    1: +math.pi / 2,  # joint2
    2: +math.pi / 2,  # joint3
    4: -math.pi / 2,  # joint5
    5: +math.pi / 2,  # joint6
}
FREE_JOINTS = (0, 3, 6)  # joint1, joint4, joint7 -> theta1, theta2, theta3

DEFAULT_TOOL_LENGTH = 0.10  # m, the drawn third link

# Everything this module draws lives under here, clear of the "mujoco" and
# "safety" prefixes `MeshcatVisualizer` publishes under.
PREFIX = "rrr"

_ARM_COLOR = 0xFFC53D  # the student's FK, drawn over the arm
_TOOL_COLOR = 0xFF5A5F  # the virtual third link
_CLOUD_COLOR = (0.32, 0.42, 0.58)  # the whole predicted workspace
_TRACE_COLOR = (1.00, 0.77, 0.24)  # the part of it the arm has visited

_JOINT_MARKER_RADIUS = 0.018
_TRIAD_SCALE = 0.12
_CLOUD_POINT_SIZE = 0.006
_TRACE_POINT_SIZE = 0.014

_PROBE_DELTA = 0.3  # rad, the step each free joint is probed with

_MIN_SPEED = 0.05  # rad/s; `set_joint_targets` raises on speed <= 0

# How far past the library's own velocity limit the sweep may be driven. That
# limit is a convention rather than a property of the robot — the MJCF carries
# no joint velocity limits at all (`xarm7_mujoco.py:59`) — and `vis2` is a
# picture, not a motion anyone is asking the hardware to make, so it is allowed
# more room. What actually stops you is the servos: the setpoint runs at
# `speed`, and past a few times the limit the arm visibly trails it.
_SPEED_HEADROOM = 4.0
# Slack added to a pose's travel time before the sweep moves on. See
# `_wait_budget`: it is what turns the speed slider into settle-vs-flow.
_SETTLE_GRACE = 0.25  # rad-equivalent


# ----------------------------------------------------------------------
# The planar arm hiding inside the xArm7
# ----------------------------------------------------------------------


def _wrap(angle):
    """Fold an angle into (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class PlanarChain:
    """The planar RRR the locked xArm7 is, and the map to and from its joints.

    Every field is measured from the loaded model by `probe`, so this holds no
    geometry of its own. For each free joint i the map is

        theta_i = sign_i * q_i + offset_i

    with `sign_i` +-1 (joints 4 and 7 turn about -z once the arm is locked, so
    theirs are negative) and `offset_i` whatever the locked pose builds in.
    """

    def __init__(self, lengths, signs, offsets, limits, heights):
        self.link_lengths = [float(v) for v in lengths]  # [l1, l2, l3]
        self._signs = np.asarray(signs, dtype=float)
        self._offsets = np.asarray(offsets, dtype=float)
        self._limits = np.asarray(limits, dtype=float)  # (3, 2), in theta
        # z of the shoulder/elbow plane and of the wrist/tool plane. The wrist
        # sits ~97 mm below the elbow, which costs the arm nothing — the three
        # axes are still parallel, so the projection onto xy is exactly planar
        # — but the drawing has to know, or the shadow floats off the arm.
        self.plane_z, self.tool_z = (float(v) for v in heights)

    # -- construction --------------------------------------------------

    @classmethod
    def probe(cls, arm, tool_length=DEFAULT_TOOL_LENGTH):
        """Measure the chain off `arm`'s MuJoCo model.

        Runs `mj_forward` on a scratch `MjData` rather than on the arm's own,
        so the simulation thread never sees the probe poses.
        """
        pose = _Prober(arm)
        home = locked_configuration()

        origins, axes, _ = pose.at(home)
        if not np.allclose(np.abs(axes[:, 2]), 1.0, atol=1e-6):
            raise RuntimeError(
                "the locked pose does not leave a planar arm: joint axes "
                f"{np.round(axes, 4).tolist()} are not all parallel to z"
            )

        shoulder, elbow, wrist = origins
        l1 = float(np.linalg.norm((elbow - shoulder)[:2]))
        l2 = float(np.linalg.norm((wrist - elbow)[:2]))

        # theta_k is segment k's heading relative to segment k-1. The headings
        # at the home pose give the offsets; stepping one joint at a time gives
        # each sign. Both are measured, neither is assumed.
        home_headings = pose.headings(home)
        signs, offsets = [], []
        for k, index in enumerate(FREE_JOINTS):
            stepped = home.copy()
            stepped[index] += _PROBE_DELTA
            moved = pose.headings(stepped)[k]
            # Only segment k and those downstream turn, so the change in the
            # relative angle is the change in this segment's own heading.
            signs.append(math.copysign(1.0, _wrap(moved - home_headings[k])))
            previous = home_headings[k - 1] if k else 0.0
            offsets.append(
                _wrap((home_headings[k] - previous) - signs[k] * home[index])
            )

        limits = []
        for k, index in enumerate(FREE_JOINTS):
            low, high = arm.position_limit[index]
            limits.append(sorted(signs[k] * np.array([low, high]) + offsets[k]))

        return cls(
            lengths=[l1, l2, float(tool_length)],
            signs=signs,
            offsets=offsets,
            limits=limits,
            heights=(shoulder[2], wrist[2]),
        )

    # -- the map -------------------------------------------------------

    def q_from_theta(self, thetas):
        """The seven joint values realizing these three planar angles.

        Clamped to the theta range the arm's joint limits allow, so a slider
        sitting on an endpoint can't round its way outside and have
        `set_joint_targets` refuse the pose.
        """
        thetas = np.clip(
            np.asarray(thetas, dtype=float).reshape(3),
            self._limits[:, 0],
            self._limits[:, 1],
        )
        q = locked_configuration()
        for k, index in enumerate(FREE_JOINTS):
            q[index] = (thetas[k] - self._offsets[k]) / self._signs[k]
        return q

    def theta_from_q(self, q):
        """The three planar angles `q` is at."""
        q = np.asarray(q, dtype=float).reshape(7)
        return np.array(
            [self._signs[k] * q[i] + self._offsets[k] for k, i in enumerate(FREE_JOINTS)]
        )

    def theta_limits(self):
        """(minJoint, maxJoint) in theta, from the arm's own joint ranges."""
        return self._limits[:, 0].copy(), self._limits[:, 1].copy()

    def vertices(self, thetas):
        """Shoulder, elbow, wrist and tool tip in world coordinates.

        Each at the height it actually sits at, so a correct FK draws itself
        onto the arm instead of hovering over it.
        """
        t1, t2, t3 = (float(v) for v in thetas)
        l1, l2, l3 = self.link_lengths
        a1, a2, a3 = t1, t1 + t2, t1 + t2 + t3
        shoulder = np.array([0.0, 0.0, self.plane_z])
        elbow = shoulder + [l1 * math.cos(a1), l1 * math.sin(a1), 0.0]
        wrist = elbow + [l2 * math.cos(a2), l2 * math.sin(a2), self.tool_z - self.plane_z]
        tip = wrist + [l3 * math.cos(a3), l3 * math.sin(a3), 0.0]
        return np.array([shoulder, elbow, wrist, tip])

    def describe(self):
        l1, l2, l3 = self.link_lengths
        return (
            f"planar RRR: l1={l1:.4f} m  l2={l2:.4f} m  l3={l3:.4f} m (drawn tool)"
            f"\n  arm plane z={self.plane_z:.3f} m, wrist plane z={self.tool_z:.3f} m"
        )


class _Prober:
    """Forward kinematics on a scratch state, for measuring the chain.

    Kept off `arm.data` on purpose: the simulation thread is reading that
    thirty times a second and the probe poses are not poses the arm is in.
    """

    def __init__(self, arm):
        self.arm = arm
        self.model = arm.model
        self.data = mujoco.MjData(arm.model)

    def at(self, q):
        """(origins, axes, tool heading) for the three free joints at `q`."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.arm._qadr] = q
        mujoco.mj_forward(self.model, self.data)

        origins, axes = [], []
        for index in FREE_JOINTS:
            jid = self.arm._jid[index]
            R = self.data.xmat[self.model.jnt_bodyid[jid]].reshape(3, 3)
            origins.append(
                self.data.xpos[self.model.jnt_bodyid[jid]]
                + R @ self.model.jnt_pos[jid]
            )
            axes.append(R @ self.model.jnt_axis[jid])

        # The drawn tool points along link7's own x axis, so that is the third
        # segment's heading.
        wrist_R = self.data.xmat[
            self.model.jnt_bodyid[self.arm._jid[FREE_JOINTS[2]]]
        ].reshape(3, 3)
        tool = math.atan2(wrist_R[1, 0], wrist_R[0, 0])
        return np.array(origins), np.array(axes), tool

    def headings(self, q):
        """World heading of each of the three planar segments at `q`, in rad."""
        origins, _, tool = self.at(q)
        return [
            math.atan2(*(origins[1] - origins[0])[[1, 0]]),
            math.atan2(*(origins[2] - origins[1])[[1, 0]]),
            tool,
        ]


def locked_configuration(thetas_free=None):
    """A seven-vector with the four held joints filled in.

    `thetas_free` is optional raw joint values for joints 1, 4 and 7.
    """
    q = np.zeros(7)
    for index, value in LOCKED_JOINTS.items():
        q[index] = value
    if thetas_free is not None:
        for k, index in enumerate(FREE_JOINTS):
            q[index] = float(thetas_free[k])
    return q


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _node(viz, path):
    return viz.viewer[f"{PREFIX}/{path}"]


def draw_polyline(viz, path, points, color=_ARM_COLOR, width=3.0):
    """Draw an open polyline through `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=np.float32).T  # meshcat wants (3, n)
    with viz.lock:
        _node(viz, path).set_object(
            g.Line(
                g.PointsGeometry(points),
                g.LineBasicMaterial(color=color, linewidth=width),
            )
        )


# Marker sets that have already been published, so a redraw only has to move
# them. These are drawn every frame, and re-sending the spheres each time would
# be a dozen ZMQ round trips per frame, taken against the same viewer lock the
# simulation thread needs to sync.
_MARKERS = {}


def draw_markers(viz, path, points, radius=_JOINT_MARKER_RADIUS, color=_ARM_COLOR):
    """Draw a small sphere at each of `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=float)
    key = (id(viz), path)
    shape = (len(points), float(radius), int(color))

    with viz.lock:
        if _MARKERS.get(key) != shape:
            _node(viz, path).delete()
            material = g.MeshLambertMaterial(color=color)
            for i in range(len(points)):
                _node(viz, f"{path}/{i}").set_object(g.Sphere(radius), material)
            _MARKERS[key] = shape
        for i, point in enumerate(points):
            T = np.eye(4)
            T[:3, 3] = point
            _node(viz, f"{path}/{i}").set_transform(T)


def draw_points(viz, path, xy, z, color=_CLOUD_COLOR, size=_CLOUD_POINT_SIZE):
    """Draw an (n, 2) set of planar points as a cloud at height `z`."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if xy.size == 0:
        return
    position = np.column_stack([xy, np.full(len(xy), float(z))]).T.astype(np.float32)
    colors = np.tile(np.asarray(color, dtype=np.float32).reshape(3, 1), (1, len(xy)))
    with viz.lock:
        _node(viz, path).set_object(
            g.PointCloud(position, colors, size=size)
        )


def draw_triad(viz, path, T, scale=_TRIAD_SCALE):
    """Draw an RGB axis triad at the 4x4 pose `T`."""
    with viz.lock:
        node = _node(viz, path)
        node.set_object(g.triad(scale))
        node.set_transform(np.asarray(T, dtype=float))


def clear(viz, path=""):
    """Remove everything this module drew under `path`."""
    with viz.lock:
        node = _node(viz, path) if path else viz.viewer[PREFIX]
        node.delete()
    for key in [k for k in _MARKERS if k[0] == id(viz) and k[1].startswith(path)]:
        del _MARKERS[key]


# ----------------------------------------------------------------------
# The student's forward kinematics, drawn over the arm
# ----------------------------------------------------------------------


class ShadowDrawer:
    """Draws whatever the student's `forward_kinematics_RRR` says the arm is.

    Held as an object only so a broken FK reports itself once instead of once
    per frame — the stub students start from returns `H_6_0 = None`, and this
    has to keep the arm usable while that is still true.
    """

    def __init__(self, viz, chain, fk_fn):
        self.viz = viz
        self.chain = chain
        self.fk_fn = fk_fn
        self._complaint = None
        self._hidden = False

    def update(self, thetas):
        """Redraw for these angles. Returns the predicted tip (x, y), or None."""
        try:
            points, tip, H_6_0 = self._solve(thetas)
        except Exception as err:
            self._complain(f"{type(err).__name__}: {err}")
            return None

        self._hidden = False
        draw_polyline(self.viz, "fk/links", points)
        draw_markers(self.viz, "fk/joints", points[:3])
        draw_polyline(self.viz, "fk/tool", points[2:], color=_TOOL_COLOR, width=5.0)

        # Placed by H_6_0 itself, not by the skeleton, so To-Do 2 is checked
        # independently of To-Do 1.
        T = np.eye(4)
        T[:2, :2] = H_6_0[:2, :2]
        T[:3, 3] = (tip[0], tip[1], self.chain.tool_z)
        draw_triad(self.viz, "fk/frame", T)
        return tip

    def _solve(self, thetas):
        """The four planar frames the student's transforms put the arm at."""
        l1, l2, l3 = self.chain.link_lengths
        result = self.fk_fn(float(thetas[0]), float(thetas[1]), float(thetas[2]), l1, l2, l3)

        if result.get("H_6_0") is None:
            raise ValueError("H_6_0 is still None (To-Do 2)")
        H_6_0 = np.asarray(result["H_6_0"], dtype=float)
        if H_6_0.shape != (3, 3):
            raise ValueError(f"H_6_0 should be 3x3, got {H_6_0.shape}")

        # The joint centres are the partial products, so this reads To-Do 1's
        # elementary transforms rather than only To-Do 2's composed answer. The
        # two are drawn separately and never reconciled: the skeleton comes
        # from the elementaries and the triad from H_6_0, so getting one right
        # and the other wrong shows up as the triad sitting off the wrist
        # instead of on it.
        keys = ("H_1_0", "H_2_1", "H_3_2", "H_4_3", "H_5_4", "H_6_5")
        H = np.eye(3)
        corners = [H.copy()]
        for key in keys:
            H = H @ np.asarray(result[key], dtype=float)
            corners.append(H.copy())

        # Frames 0, 2 and 4 are the three joints; frame 6 is the tool tip.
        planar = [corners[i][:2, 2] for i in (0, 2, 4, 6)]
        heights = (
            self.chain.plane_z,
            self.chain.plane_z,
            self.chain.tool_z,
            self.chain.tool_z,
        )
        points = np.array([[p[0], p[1], z] for p, z in zip(planar, heights)])
        return points, (float(H_6_0[0, 2]), float(H_6_0[1, 2])), H_6_0

    def _complain(self, message):
        if not self._hidden:
            self._hidden = True
            clear(self.viz, "fk")
        if message == self._complaint:
            return
        self._complaint = message
        print(
            f"[vis] forward_kinematics_RRR isn't usable yet — {message}\n"
            "      The arm still moves; the predicted skeleton stays hidden "
            "until the To-Dos in ex1/forward_kinematics_RRR.py are filled in."
        )


# ----------------------------------------------------------------------
# Joint limits the arm can actually sweep
# ----------------------------------------------------------------------


def _one_turn(low, high):
    """Trim a joint range spanning more than 2*pi down to one revolution."""
    if high - low <= 2.0 * math.pi:
        return low, high
    middle = 0.5 * (low + high)
    return middle - math.pi, middle + math.pi


def collision_free_limits(arm, chain, samples=200, wrist_samples=12, margin=0.05):
    """(minJoint, maxJoint) in theta, with self-colliding elbow angles removed.

    theta1 and theta3 are left at the arm's full range: turning the base cannot
    create a self-collision, and the wrist is deliberately unconstrained. Only
    theta2 needs narrowing, and it is narrowed by asking the arm's own guard
    (`check_safety`, the same check that stops the simulated arm) rather than
    by picking a number: scan theta2 across its range, keep the longest run
    that is clear at every wrist angle, then inset by `margin`.

    Checking every wrist angle matters. The forearm folded back over the upper
    link is a collision whatever the wrist does, but nearer the edge of the
    band it is the wrist that touches first, and a band validated at one wrist
    angle would hand the sweep poses the guard then refuses.
    """
    low, high = chain.theta_limits()
    # joint1 and joint7 both turn through more than a full revolution, and a
    # sweep past one adds no workspace — it re-traces what it already drew, at
    # twice the cost. theta2 has well under a turn to give and is left alone.
    low[0], high[0] = _one_turn(low[0], high[0])
    low[2], high[2] = _one_turn(low[2], high[2])

    grid = np.linspace(low[1], high[1], int(samples))
    wrists = np.linspace(0.0, 2.0 * math.pi, int(wrist_samples), endpoint=False)

    safe = np.array(
        [
            all(arm.check_safety(chain.q_from_theta([0.0, t2, t3])) is None for t3 in wrists)
            for t2 in grid
        ]
    )

    start, best = None, None
    for i, ok in enumerate(np.append(safe, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None:
        raise RuntimeError(
            "no self-collision-free elbow angle found; the locked pose may be wrong"
        )

    lo, hi = grid[best[0]], grid[best[1] - 1]
    lo, hi = lo + margin, hi - margin
    if lo >= hi:
        raise RuntimeError(f"the collision-free elbow band is narrower than {margin} rad")

    print(
        f"[vis] elbow band, self-collision free: theta2 in "
        f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg "
        f"(of [{math.degrees(low[1]):.1f}, {math.degrees(high[1]):.1f}] reachable)"
    )
    return (
        [float(low[0]), float(lo), float(low[2])],
        [float(high[0]), float(hi), float(high[2])],
    )


# ----------------------------------------------------------------------
# The slider page
# ----------------------------------------------------------------------

# `string.Template` rather than `str.format`: the page is mostly CSS and JS, and
# `.format` would need every brace in it doubled.
_PAGE = string.Template("""<!doctype html>
<title>$title</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #14171c; color: #e8ecf1; }
  #panel { padding: 10px 16px 12px; border-bottom: 1px solid #2b313a; }
  .row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
  .row label { width: 5.5em; color: #9aa7b8; }
  .row input { flex: 1; }
  .row output { width: 7em; text-align: right; font-variant-numeric: tabular-nums; }
  #status { margin-top: 8px; color: #9aa7b8; font-variant-numeric: tabular-nums; }
  iframe { display: block; width: 100vw; height: calc(100vh - $chrome); border: 0; }
  button { background: #2b313a; color: #e8ecf1; border: 0; padding: 4px 12px;
           border-radius: 4px; cursor: pointer; }
</style>
<div id="panel">
$rows
  <div class="row">
    <div id="status">&mdash;</div>
    <button id="reset">reset</button>
  </div>
</div>
<iframe src="$meshcat_url"></iframe>
<script>
  const SPECS = $specs;
  const el = k => document.getElementById(k);

  function show(spec) {
    const v = (+el("s_" + spec.key).value).toFixed(spec.digits);
    el("o_" + spec.key).textContent = v + spec.unit;
  }
  function push() {
    SPECS.forEach(show);
    fetch("set?" + SPECS.map(s => s.key + "=" + el("s_" + s.key).value).join("&"));
  }
  SPECS.forEach(s => el("s_" + s.key).addEventListener("input", push));
  el("reset").addEventListener("click", () => {
    SPECS.forEach(s => el("s_" + s.key).value = s.value);
    push();
  });

  setInterval(async () => {
    const state = await (await fetch("state")).json();
    el("status").innerHTML = state.status;
  }, 250);
  push();
</script>
""")

_ROW = string.Template(
    '  <div class="row">\n'
    '    <label for="s_$key">$label</label>\n'
    '    <input id="s_$key" type="range" min="$low" max="$high" step="$step" value="$value">\n'
    '    <output id="o_$key"></output>\n'
    "  </div>"
)

# Page chrome above the viewer: one row per slider, plus the status row.
_ROW_HEIGHT = 30  # px
_PANEL_PADDING = 40  # px


class SliderSpec(collections.namedtuple(
    "SliderSpec", "key label low high step value unit digits"
)):
    """One slider on the page.

    `unit` and `digits` are only how the browser renders the number back to the
    reader; python always works in the raw value.
    """

    __slots__ = ()

    def __new__(cls, key, label, low, high, step, value, unit="", digits=2):
        return super().__new__(
            cls, key, label, float(low), float(high), float(step),
            float(np.clip(value, low, high)), unit, int(digits),
        )


class SliderPage:
    """Sliders on a local page, with the meshcat viewer embedded below.

    meshcat's own dat.GUI can't be driven from python — its ZMQ bridge forwards
    only the five scene commands and drops anything else, and the browser->python
    direction is wired up for image captures alone — so the controls are served
    from here instead, and the viewer is put in an iframe so it is still one tab.

    Values live behind a lock: `values` is what the caller reads each tick, and
    `set_status` is the line the page echoes back. The page knows nothing about
    what the numbers mean; formatting the status line is the caller's job.
    """

    def __init__(self, specs, meshcat_url, title="xArm7", open_browser=True):
        self._specs = list(specs)
        self._lock = threading.Lock()
        self._values = {s.key: s.value for s in self._specs}
        self._status = "&mdash;"

        page = _PAGE.substitute(
            title=title,
            meshcat_url=meshcat_url,
            chrome=f"{_PANEL_PADDING + _ROW_HEIGHT * (len(self._specs) + 1)}px",
            rows="\n".join(
                _ROW.substitute(
                    key=s.key, label=s.label,
                    low=f"{s.low:.4f}", high=f"{s.high:.4f}",
                    step=f"{s.step:g}", value=f"{s.value:.4f}",
                )
                for s in self._specs
            ),
            specs=json.dumps([
                {"key": s.key, "value": s.value, "unit": s.unit, "digits": s.digits}
                for s in self._specs
            ]),
        ).encode("utf-8")

        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", _free_port()), _handler_for(self, page)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="rrr-sliders", daemon=True
        )
        self._thread.start()

        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/"
        print(f"[vis] controls: {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    @property
    def values(self):
        """The current slider values, by key."""
        with self._lock:
            return dict(self._values)

    def set_status(self, text):
        """Publish the line the page shows under the sliders."""
        with self._lock:
            self._status = str(text)

    def _accept(self, query):
        parsed = urllib.parse.parse_qs(query)
        with self._lock:
            for spec in self._specs:
                value = parsed.get(spec.key)
                if value:
                    self._values[spec.key] = float(np.clip(
                        float(value[0]), spec.low, spec.high
                    ))

    def _state(self):
        with self._lock:
            return {"values": dict(self._values), "status": self._status}

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _handler_for(sliders, page):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            route, _, query = self.path.lstrip("/").partition("?")
            if route in ("", "index.html"):
                self._reply("text/html; charset=utf-8", page)
            elif route == "set":
                sliders._accept(query)
                self._reply("text/plain", b"ok")
            elif route == "state":
                body = json.dumps(sliders._state()).encode("utf-8")
                self._reply("application/json", body)
            else:
                self.send_error(404)

        def _reply(self, content_type, body):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # a slider drag is a request per frame; don't narrate it

    return Handler


# ----------------------------------------------------------------------
# Shared setup
# ----------------------------------------------------------------------


def start_arm(tool_length=DEFAULT_TOOL_LENGTH, open_browser=True, **kwargs):
    """A visualized xArm7 holding the locked pose, and its `PlanarChain`.

    The safety box is off: it is drawn around the volume the *lab* arm may use,
    and the planar sweep turns the whole way round, so leaving it on would clip
    most of theta1 away. Self-collision checking stays on — that is the
    constraint the exercises care about.

    The visualizer is attached here rather than through `visualize=True` only
    so it can be told not to open a tab of its own: `visualize_ex1` serves the
    viewer inside its own page, and two tabs of the same scene is one too many.
    """
    options = dict(visualize=False, safety_box=None, guard=True)
    options.update(kwargs)
    arm = SimulatedXArm7(**options)
    try:
        arm.viz = MeshcatVisualizer(arm.model, arm.data, open_browser=open_browser)
        print(f"[vis] meshcat: {arm.viz.url}")
        chain = PlanarChain.probe(arm, tool_length=tool_length)
        print("[vis] " + chain.describe())
        arm.set_joint_targets(locked_configuration(), speed=0.8)
    except Exception:
        arm.close()
        raise
    return arm, chain


def tool_segment(arm, chain):
    """The virtual third link where the arm is right now: (wrist, tip).

    Read from `data` under the arm's lock, since the simulation thread is
    writing it. The tool has no geometry in the model — it is drawn, not
    simulated — so it is placed on link7's own x axis.
    """
    with arm._lock:
        jid = arm._jid[FREE_JOINTS[2]]
        body = arm.model.jnt_bodyid[jid]
        R = arm.data.xmat[body].reshape(3, 3)
        wrist = arm.data.xpos[body] + R @ arm.model.jnt_pos[jid]
    return wrist, wrist + chain.link_lengths[2] * R[:, 0]


def measured_tip(arm, chain):
    """Where the drawn tool tip actually is, from MuJoCo. (x, y)."""
    tip = tool_segment(arm, chain)[1]
    return float(tip[0]), float(tip[1])


def draw_tool(viz, arm, chain):
    """Draw the virtual third link out of the flange, where the arm is now."""
    wrist, tip = tool_segment(arm, chain)
    draw_polyline(viz, "tool", np.array([wrist, tip]), color=_TOOL_COLOR, width=6.0)
    draw_markers(viz, "tool_tip", np.array([tip]), radius=0.012, color=_TOOL_COLOR)


# ----------------------------------------------------------------------
# ex1: forward kinematics, on sliders
# ----------------------------------------------------------------------


def visualize_ex1(fk_fn, rate=30.0, tool_length=DEFAULT_TOOL_LENGTH, open_browser=True):
    """Drive the arm from three sliders, drawing the student's FK over it.

    The sliders drive the *arm* directly through the calibrated map, not
    through `fk_fn` — so the demo works from the very first run, with the
    exercise still untouched. `fk_fn` only draws the skeleton it predicts. When
    the two coincide the drawn arm sits inside the real one; when they don't,
    the gap is the error.
    """
    # The viewer goes inside the slider page, so it doesn't open a tab itself.
    arm, chain = start_arm(tool_length=tool_length, open_browser=False)
    page = None
    try:
        low, high = chain.theta_limits()
        page = SliderPage(
            [
                SliderSpec(
                    key=f"t{i}", label=f"theta{i + 1}",
                    low=low[i], high=high[i], step=0.005, value=0.0,
                    unit="\u00b0", digits=1,
                )
                for i in range(3)
            ],
            arm.viz.url,
            title="RRR forward kinematics \u2014 xArm7",
            open_browser=open_browser,
        )
        shadow = ShadowDrawer(arm.viz, chain, fk_fn)
        print("[vis] drag the sliders; ctrl-c here to stop.")

        period = 1.0 / float(rate)
        while True:
            values = page.values
            thetas = np.array([values[f"t{i}"] for i in range(3)])
            arm.servo_joints(chain.q_from_theta(thetas))
            draw_tool(arm.viz, arm, chain)
            page.set_status(
                f"your FK tip <b>{_xy(shadow.update(thetas))}</b> &nbsp; "
                f"measured tip <b>{_xy(measured_tip(arm, chain))}</b>"
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()


def _xy(point):
    """A tip position for the status line, or a dash if there isn't one."""
    if point is None:
        return "\u2014"
    return f"({point[0]:.3f}, {point[1]:.3f})"


# ----------------------------------------------------------------------
# ex2: the workspace, swept
# ----------------------------------------------------------------------


def visualize_ex2(
    workspace_fn,
    ee_fn=None,
    cloud_samples=25,
    walk_samples=6,
    speed=1.2,
    max_speed=None,
    tool_length=DEFAULT_TOOL_LENGTH,
    open_browser=True,
):
    """Draw the student's predicted workspace, then walk the arm through it.

    `workspace_fn` is called once, over the collision-free joint limits, and
    everything it returns is drawn at once as the faint cloud — that is the
    student's own answer, on the robot's own scale. The arm then visits a
    coarser grid over the same limits in the same nested order the exercise
    sweeps in (theta1 outermost, theta3 innermost), lighting up each point as
    it arrives, so the cloud fills in the way the loops produce it.

    A speed slider sets the pace, live: `speed` is only where it starts and
    `max_speed` how far it can be pushed. See `_wait_budget` for what the top of
    the range does, and `raise_speed_ceiling` for why it may exceed the arm's
    nominal 3.14 rad/s.
    """
    # The viewer goes inside the control page, so it doesn't open a tab itself.
    arm, chain = start_arm(tool_length=tool_length, open_browser=False)
    page = None
    try:
        ceiling = raise_speed_ceiling(arm, max_speed)
        minJoint, maxJoint = collision_free_limits(arm, chain)
        page = SliderPage(
            [SliderSpec(
                key="speed", label="speed",
                # The floor stays off zero because `set_joint_targets` raises
                # on speed <= 0; the ceiling is whatever `raise_speed_ceiling`
                # just allowed.
                low=_MIN_SPEED, high=ceiling,
                step=0.05, value=speed, unit=" rad/s", digits=2,
            )],
            arm.viz.url,
            title="RRR workspace sweep \u2014 xArm7",
            open_browser=open_browser,
        )

        try:
            xs, ys = workspace_fn(
                nSamples=int(cloud_samples),
                minJoint=list(minJoint),
                maxJoint=list(maxJoint),
                linkLengths=list(chain.link_lengths),
            )
            cloud = np.column_stack(
                [np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)]
            )
            print(f"[vis] workspace_analysis_RRR returned {len(cloud)} points.")
            draw_points(arm.viz, "workspace", cloud, chain.tool_z)
        except Exception as err:
            # The shipped stub raises. The walk below is still worth watching,
            # and it still traces a workspace — from the calibrated chain
            # rather than from the student's answer.
            print(
                f"[vis] workspace_analysis_RRR isn't usable yet — "
                f"{type(err).__name__}: {err}\n"
                "      Sweeping anyway: the arm traces the workspace as it "
                "goes, and your own predicted cloud will appear behind it once "
                "the To-Dos in ex2/workspace_analysis_RRR.py are filled in."
            )

        grids = [
            np.linspace(minJoint[i], maxJoint[i], int(walk_samples)) for i in range(3)
        ]
        total = int(walk_samples) ** 3
        print(f"[vis] walking {total} configurations; drag `speed` to hurry it "
              "along, ctrl-c here to stop.")

        started = time.perf_counter()
        traced = []
        visited = 0
        for t1 in grids[0]:
            for t2 in grids[1]:
                for t3 in grids[2]:
                    thetas = (t1, t2, t3)
                    q = chain.q_from_theta(thetas)
                    if arm.check_safety(q) is not None:
                        continue  # the derived band should make this unreachable
                    # Re-read every pose, so dragging the slider takes effect on
                    # the next move — the call below blocks, so it cannot land
                    # any sooner than that.
                    pace = page.values["speed"]
                    arm.set_joint_targets(
                        q, speed=pace, wait=True,
                        timeout=_wait_budget(arm, q, pace),
                    )
                    draw_tool(arm.viz, arm, chain)

                    traced.append(_traced_point(ee_fn, chain, thetas))
                    draw_points(
                        arm.viz, "traced", np.array(traced), chain.tool_z,
                        color=_TRACE_COLOR, size=_TRACE_POINT_SIZE,
                    )
                    visited += 1
                    page.set_status(
                        f"pose <b>{visited}/{total}</b> &nbsp; "
                        f"<b>{pace:.2f}</b> rad/s &nbsp; "
                        f"<b>{time.perf_counter() - started:.0f}</b>s"
                    )
        print(f"[vis] done: {visited}/{total} configurations reached.")
        arm.set_joint_targets(locked_configuration(), speed=page.values["speed"])
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()


def raise_speed_ceiling(arm, max_speed=None):
    """Let the sweep command joint speeds above the library's limit.

    `set_joint_targets` clips `speed` to `arm.velocity_limit`, so raising the
    slider's ceiling without raising that too would do nothing at all — the
    number on the page would climb and the arm would keep moving at 3.14 rad/s.

    That limit is `xarm7_mujoco.VELOCITY_LIMIT`, a flat 3.14 rad/s the library
    picks because the MJCF declares none. It is the speed the *real* arm is
    held to, and nothing here goes near the real arm: this is a visualization
    of a workspace, and waiting out a 216-pose sweep at walking pace is the
    only thing the limit accomplishes.

    Raised only on this arm object, only for the sweep. Returns the ceiling.
    """
    nominal = float(np.max(arm.velocity_limit))
    ceiling = nominal * _SPEED_HEADROOM if max_speed is None else float(max_speed)
    ceiling = max(ceiling, _MIN_SPEED)
    arm.velocity_limit = np.full_like(arm.velocity_limit, ceiling)
    if ceiling > nominal:
        print(f"[vis] speed ceiling raised to {ceiling:.2f} rad/s "
              f"({ceiling / nominal:.1f}x the arm's nominal {nominal:.2f}); "
              "the servos will trail the setpoint near the top.")
    return ceiling


def _wait_budget(arm, q, speed):
    """How long to let one grid pose have before moving on, in seconds.

    At the bottom of the speed range this is comfortably longer than the move,
    so the arm settles at every pose exactly as it used to. As the slider rises
    the grace shrinks and the next pose is commanded before the arm has stopped,
    so the sweep flows continuously instead of stuttering to a halt 216 times.
    `set_joint_targets` re-plans from wherever the arm actually is, so cutting a
    move short this way is safe and costs no accuracy — the traced points come
    from the commanded angles, not from the arm.

    The grace divides by speed rather than being a constant so that the two
    regimes are one expression: 0.8 s of slack at 0.3 rad/s, 0.08 s at 3.14.
    """
    span = float(np.max(np.abs(q - arm.joint_values)))
    return (span + _SETTLE_GRACE) / float(speed)


def _traced_point(ee_fn, chain, thetas):
    """The point to light up: the student's, or the reference chain's.

    The fallback is the calibrated chain at the *commanded* angles rather than
    the arm's measured tip. It is the same point to ~5e-16 m, and it stays right
    when the sweep is running fast enough that the arm is still moving when the
    next pose is commanded — a measured tip read mid-move would smear the cloud.
    """
    if ee_fn is not None:
        try:
            x, y = ee_fn(list(thetas), list(chain.link_lengths))
            return [float(x), float(y)]
        except Exception:
            pass  # endEffector_RRR is still a stub; fall back to the reference
    return [float(v) for v in chain.vertices(thetas)[3][:2]]
