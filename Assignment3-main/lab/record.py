"""Send the arm to the planar pose, then record hand-guided RR paths.

    python record.py                    # into recordings/rr-<timestamp>.csv
    python record.py my-circle.csv      # or wherever you say

The robot starts from its current position, and first squares the locked
joints (2, 3, 5, 6). If that move is refused, use
        `python -m xarm7_lib.free_drive`
to move the arm somewhere legal, then run this script again.

Then this robot will enter joint teaching mode, and the arm is pushed by hand.
Joints 2, 3, 5 and 6 are watched and put back if they drift. A live plot shows
the RR arm and the path it has been through, with the end effector's velocity
as an arrow off its tip. Ctrl-c ends the recording, and the path is saved.

Each recording is a plain csv of the RR joint angles, which can be replayed
with `replay.py`.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import argparse
import select
import termios
import signal
import math
import sys
import os

from xarm7_lib import RealXArm7, Robot
from xarm7_lib.free_drive import FREE_JOINTS

from live_plot import LivePlot
from robot_info import LOCKED_ANGLES_DEG, LOCKED_INDICES, RECORD_RATE, q2rr, qd2rr


class CtrlC:
    """Ctrl-c as a request to finish, noticed at the next safe point."""
    def __init__(self):
        self.requested = False

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\n[record] ctrl-c: finishing (again to force)")

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)


def reset(arm):
    """Move the arm to the locked pose, leaving the free joints wherever they are.
    The locked joints are 2, 3, 5 and 6 at +90, +90, -90 and +90 degrees.
    """
    start = arm.joint_values
    locked = start.copy()
    locked[LOCKED_INDICES] = np.radians(LOCKED_ANGLES_DEG)
    if not arm.set_joint_targets(locked):
        raise RuntimeError("Arm never reached the locked pose.")


def save_recording(traj):
    """A free-drive `Trajectory` as the file's (N, 2) angles, one row per tick."""
    theta = np.column_stack(q2rr(traj.q.T))
    changed = np.any(np.diff(theta, axis=0) != 0, axis=1)
    reports = np.flatnonzero(np.concatenate([[True], changed]))
    ticks = np.arange(len(theta))
    return np.column_stack(
        [np.interp(ticks, reports, column[reports]) for column in theta.T]
    )


def main(arm: Robot, out, plot, ctrl_c):
    """One free drive: record until ctrl-c, then save."""
    plot.reset("recording — ctrl-c to stop")
    print("[record] free drive: push the arm through the path to record.\n"
          "       Ctrl-c stops it there and saves what it has been through.")

    def on_sample(t, q):
        # The measured joint speeds, read next to the angles the loop just took,
        # so the arrow is the arm's velocity as it is being pushed.
        plot.update(q2rr(q), qd2rr(arm.joint_velocities))
        return ctrl_c.requested  # True ends the run, and the arm holds where it is

    def pause_until_safe(message):
        """Hold the recovery move until hands are clear. False ends the run.

        The library's own prompt blocks in `input()`, which would sit on a
        ctrl-c until someone pressed enter. Poll instead, so ctrl-c is noticed
        and the plot keeps its window alive while it waits. A closed stdin
        means nobody is at the terminal to say when it is safe, so the run
        ends rather than moving the arm at them.
        """
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)  # only a fresh enter counts
        print(message)
        print("        press enter when your hands are clear: ", end="", flush=True)
        while not ctrl_c.requested:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return sys.stdin.readline() != ""
            plot.idle()
        return False

    # math.inf: the run ends when `on_sample` says so, not on a clock.
    traj = arm.free_drive(FREE_JOINTS, duration=math.inf,
                          tolerance=0.1,
                          on_sample=on_sample,
                          confirm=pause_until_safe)

    theta = save_recording(traj)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, theta, delimiter=",", fmt="%.10f",
               header=f"theta1,theta2 in radians, {RECORD_RATE:.0f} Hz")
    n = len(theta)
    print(f"[record] {n} samples over {n / RECORD_RATE:.1f}s written to {out}")
    plot.draw(f"saved {out.name} — {n} samples", force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out", nargs="?",
        help="file to write the recording to "
        "(default: recordings/rr-<timestamp>.csv)",
    )
    return parser.parse_args()

def default_recording_path():
    """recordings/rr-<timestamp>.csv, stamped when free drive starts.

    One run records once and takes longer than a second to do it, so two of
    them can never land on the same name.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("recordings") / f"rr-{stamp}.csv"

if __name__ == "__main__":
    args = parse_args()
    np.set_printoptions(precision=3, suppress=True)

    if not os.environ.get("ROBOT_IP", "").strip():
        raise SystemExit(
            "[record] ROBOT_IP is not set, so there is no arm to hand-guide.\n"
            "       Set it to the controller's address and run again."
        )

    with Robot() as robot:
        if not isinstance(robot.robot, RealXArm7):
            raise SystemExit("[record] Robot() gave the simulation, not the real arm.")

        reset(robot)
        out = Path(args.out) if args.out else default_recording_path()
        with CtrlC() as ctrl_c:
            main(robot, out, LivePlot(), ctrl_c)
