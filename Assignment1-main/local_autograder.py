"""Autograder for HW1: forward_kinematics_RRR (ex1) and
workspace_analysis_PR / workspace_analysis_RRR (ex2).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TOL = 1e-4

MAX_FAILURES_SHOWN = 10
MAX_DETAIL_CHARS = 160


def _load_module(path):
    """Import a standalone script (no package, no __init__.py) by path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path):
    """Stream one JSON object per line.

    A generator, not a list: ex1's fixtures run to millions of cases, and
    holding both the inputs and the expected transforms in memory at once does
    not fit. Every caller only iterates, in step with `zip`.
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _close(actual, expected, tol=TOL):
    """np.allclose that fails (rather than raising) on garbage/missing values."""
    try:
        return bool(np.allclose(
            np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
            atol=tol, rtol=0,
        ))
    except (TypeError, ValueError):
        return False


def _sorted_points(xs, ys):
    """(x, y) pairs, rounded and order-independent.

    The loop order a student sweeps joints in isn't part of the spec — only
    the resulting workspace shape is — so points are compared as a set
    rather than position-by-position.
    """
    return np.array(sorted(zip(np.round(xs, 6), np.round(ys, 6))))


class Grader:
    """Counts cases and keeps only the failures it is going to print.

    Storing a row per case is what a few thousand fixtures allow and a few
    million do not, so this holds running totals and a bounded list. `keep` is
    how many failures it retains; `--verbose` raises it rather than switching
    the accounting off.
    """

    def __init__(self, name, keep=MAX_FAILURES_SHOWN):
        self.name = name
        self.passed = 0
        self.total = 0
        self.keep = keep
        self.failures = []  # (label, detail), capped at `keep`

    def record(self, label, passed, detail=""):
        self.total += 1
        if passed:
            self.passed += 1
        elif len(self.failures) < self.keep:
            self.failures.append((label, detail))

    def report(self, max_failures=MAX_FAILURES_SHOWN):
        """Print the result: a passing run is one line, a failing one lists the
        failures it kept and says how many it didn't."""
        print(f"\n=== {self.name}: {self.passed}/{self.total} ===")

        shown = min(max_failures, len(self.failures))
        for label, detail in self.failures[:shown]:
            line = f"  [FAIL] {label}"
            if detail:
                if len(detail) > MAX_DETAIL_CHARS:
                    detail = detail[:MAX_DETAIL_CHARS] + "..."
                line += f" — {detail}"
            print(line)

        hidden = (self.total - self.passed) - shown
        if hidden > 0:
            print(f"  ... and {hidden} more failure(s); pass --verbose to keep more")
        return self.passed, self.total


def grade_ex1(keep=MAX_FAILURES_SHOWN):
    grader = Grader("ex1: forward_kinematics_RRR", keep)
    in_path, out_path = ROOT / "ex1" / "in1.txt", ROOT / "ex1" / "out1.txt"
    if not in_path.exists() or not out_path.exists():
        grader.record("fixtures present", False, f"missing {in_path} or {out_path}")
        return grader

    try:
        module = _load_module(ROOT / "ex1" / "forward_kinematics_RRR.py")
    except Exception as err:
        grader.record("module imports", False, repr(err))
        return grader

    cases = _read_jsonl(in_path)
    expected_cases = _read_jsonl(out_path)
    elementary = ["H_1_0", "H_2_1", "H_3_2", "H_4_3", "H_5_4", "H_6_5"]

    for i, (case, expected) in enumerate(zip(cases, expected_cases)):
        label = (
            f"case {i} (theta=[{case['theta1']:.2f}, {case['theta2']:.2f}, "
            f"{case['theta3']:.2f}])"
        )
        try:
            actual = module.forward_kinematics_RRR(
                case["theta1"], case["theta2"], case["theta3"],
                case["l1"], case["l2"], case["l3"],
            )
        except Exception as err:
            grader.record(f"{label} To-Do 1 (elementary transforms)", False, repr(err))
            grader.record(f"{label} To-Do 2 (H_6_0)", False, repr(err))
            continue

        elem_ok = all(
            key in actual and _close(actual[key], expected[key]) for key in elementary
        )
        grader.record(
            f"{label} To-Do 1 (elementary transforms)", elem_ok,
            "" if elem_ok else "H_1_0..H_6_5 don't all match the expected transforms",
        )

        h60_ok = "H_6_0" in actual and _close(actual["H_6_0"], expected["H_6_0"])
        grader.record(
            f"{label} To-Do 2 (H_6_0)", h60_ok,
            "" if h60_ok else f"got {actual.get('H_6_0')}, want {expected['H_6_0']}",
        )

    return grader


def grade_workspace(label, fn_name, module_path, in_path, out_path,
                    keep=MAX_FAILURES_SHOWN):
    grader = Grader(label, keep)
    if not in_path.exists() or not out_path.exists():
        grader.record("fixtures present", False, f"missing {in_path} or {out_path}")
        return grader

    try:
        module = _load_module(module_path)
    except Exception as err:
        grader.record("module imports", False, repr(err))
        return grader

    fn = getattr(module, fn_name, None)
    if fn is None:
        grader.record(f"{fn_name} defined", False, "function not found in module")
        return grader

    cases = _read_jsonl(in_path)
    expected_cases = _read_jsonl(out_path)

    for i, (case, expected) in enumerate(zip(cases, expected_cases)):
        case_label = f"case {i} (nSamples={case['nSamples']})"
        try:
            xs, ys = fn(
                nSamples=case["nSamples"],
                minJoint=case["minJoint"],
                maxJoint=case["maxJoint"],
                linkLengths=case["linkLengths"],
            )
            xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        except Exception as err:
            grader.record(case_label, False, repr(err))
            continue

        want_xs, want_ys = np.asarray(expected["xs"]), np.asarray(expected["ys"])
        if xs.size != want_xs.size:
            grader.record(case_label, False, f"expected {want_xs.size} points, got {xs.size}")
            continue

        ok = _close(_sorted_points(xs, ys), _sorted_points(want_xs, want_ys))
        grader.record(case_label, ok, "" if ok else "workspace point set doesn't match")

    return grader


def _meshcat_vis():
    """Import the visualization module, or explain what's missing.

    The graded modes need only numpy, so a student can be grading happily and
    still be one `conda env create` short of the simulator.
    """
    try:
        import meshcat_vis
    except ImportError as err:
        sys.exit(
            f"Visualization needs the simulator, and it isn't importable: {err}\n"
            "Create the environment it lives in with:\n"
            "    conda env create -f xarm7_lib/environment.yml && conda activate 16384"
        )
    return meshcat_vis


def visualize_ex1():
    """Drive the simulated xArm7 from sliders, drawing the student's ex1 FK."""
    module = _load_module(ROOT / "ex1" / "forward_kinematics_RRR.py")
    # module = _load_module(ROOT / "solutions" / "ref_solution_ex1.py")
    _meshcat_vis().visualize_ex1(module.forward_kinematics_RRR)
    


def visualize_ex2(cloud_samples, walk_samples, speed, max_speed=None):
    """Draw the student's ex2 workspace and sweep the arm through it."""
    module = _load_module(ROOT / "ex2" / "workspace_analysis_RRR.py")
    # module = _load_module(ROOT / "solutions" / "ref_solution_ex2.py")
    _meshcat_vis().visualize_ex2(
        module.workspace_analysis_RRR,
        getattr(module, "endEffector_RRR", None),
        cloud_samples=cloud_samples,
        walk_samples=walk_samples,
        speed=speed,
        max_speed=max_speed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--problem", type = str, choices=["ex1", "ex2", "vis1", "vis2", "all"], default="all",)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every test case, not just failures")
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES_SHOWN,
                        help="failures to list per exercise (default: %(default)s)")
    parser.add_argument("--cloud-samples", type=int, default=25,
                        help="vis2: grid resolution of the drawn workspace "
                             "(default: %(default)s)")
    parser.add_argument("--walk-samples", type=int, default=6,
                        help="vis2: grid resolution the arm actually visits, "
                             "cubed (default: %(default)s)")
    parser.add_argument("--speed", type=float, default=7.0,
                        help="vis2: starting joint speed in rad/s; the page has "
                             "a slider to change it live (default: %(default)s)")
    parser.add_argument("--max-speed", type=float, default=7.5,
                        help="vis2: top of the speed slider in rad/s "
                             "(default: 4x the arm's nominal 3.14)")
    args = parser.parse_args()

    # The visualization modes show, they don't score; they never reach the
    # report below.
    if args.problem == "vis1":
        visualize_ex1()
        return
    if args.problem == "vis2":
        visualize_ex2(args.cloud_samples, args.walk_samples, args.speed,
                      args.max_speed)
        return

    # --verbose keeps every failure rather than switching the accounting off:
    # at a few million cases, storing them all is what has to stay bounded.
    keep = 10 ** 9 if args.verbose else args.max_failures

    if(args.problem == "all"):
        graders = [
            grade_ex1(keep),
            grade_workspace(
                "ex2: workspace_analysis_PR", "workspace_analysis_PR",
                ROOT / "ex2" / "workspace_analysis_PR.py",
                ROOT / "ex2" / "in1.txt", ROOT / "ex2" / "out1.txt", keep,
            ),
            grade_workspace(
                "ex2: workspace_analysis_RRR", "workspace_analysis_RRR",
                ROOT / "ex2" / "workspace_analysis_RRR.py",
                ROOT / "ex2" / "in2.txt", ROOT / "ex2" / "out2.txt", keep,
            ),
        ]
    elif(args.problem == "ex1"):
            graders = [
                grade_ex1(keep),
            ]
    elif(args.problem == "ex2"):
        graders = [
            grade_workspace(
                "ex2: workspace_analysis_PR", "workspace_analysis_PR",
                ROOT / "ex2" / "workspace_analysis_PR.py",
                ROOT / "ex2" / "in1.txt", ROOT / "ex2" / "out1.txt", keep,
            ),
            grade_workspace(
                "ex2: workspace_analysis_RRR", "workspace_analysis_RRR",
                ROOT / "ex2" / "workspace_analysis_RRR.py",
                ROOT / "ex2" / "in2.txt", ROOT / "ex2" / "out2.txt", keep,
            ),
        ]

    total_passed = total_count = 0
    for grader in graders:
        p, t = grader.report(max_failures=args.max_failures)
        total_passed += p
        total_count += t

    print(f"\n=== TOTAL: {total_passed}/{total_count} ===")
    sys.exit(0 if total_passed == total_count else 1)


if __name__ == "__main__":
    main()
