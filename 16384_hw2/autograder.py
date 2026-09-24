"""Self-check for Assignment 2.  Run this before you submit:

    python autograder.py

It runs your functions over the recorded data in common/data.csv and compares
the results against the reference answers in common/solution.csv.  For anything
that does not match it prints the input, what was expected, and what you gave.

This is a convenience tool, not the grader.  The Gradescope autograder runs the
same comparisons plus additional randomly generated configurations.
"""

import os
import pickle
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, 'ex1'), os.path.join(ROOT, 'ex2')):
    if p not in sys.path:
        sys.path.insert(0, p)

TOL = 1e-6
STRIDE = 5          # check every Nth row of data.csv
POINTS = ['COM_Link_1', 'End_Link_1', 'COM_Link_2', 'End_Link_2']
DESIRED_FORCE = np.array([-1.0, 1.0])
GRAVITY = np.array([0.0, -9.8])


def vec(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return f'{values: .6f}'
    return '[' + ', '.join(f'{v: .6f}' for v in values.reshape(-1)) + ']'


def block(rows, indent='    '):
    width = max(len(label) for label, _ in rows) if rows else 0
    return '\n'.join(f'{indent}{label:<{width}} = {vec(value)}' for label, value in rows)


class Checker:
    def __init__(self):
        self.failed = 0
        self.passed = 0
        self.skipped = 0

    def report_pass(self, name):
        self.passed += 1
        print(f'[ OK ] {name}')

    def report_fail(self, name, message):
        self.failed += 1
        print(f'[FAIL] {name}')
        print(message)

    def report_skip(self, name, why):
        self.skipped += 1
        print(f'[SKIP] {name}  ({why})')

    def compare(self, name, labels, got, expected, inputs):
        """Compare per-row outputs and show the first row that disagrees.

        ``got`` and ``expected`` are (n_rows, n_labels, ...) arrays; ``inputs``
        is a list of (label, per-row-value-or-constant) pairs to print.
        """
        got = np.asarray(got, dtype=float)
        expected = np.asarray(expected, dtype=float)

        if got.shape != expected.shape:
            self.report_fail(name, f'    expected shape {expected.shape}, '
                                   f'got shape {got.shape}')
            return

        # np.isclose is False for NaN and inf, so those land in `bad` too.
        rest = tuple(range(1, got.ndim))
        bad = ~np.all(np.isclose(got, expected, rtol=0, atol=TOL), axis=rest)
        if not bad.any():
            self.report_pass(name)
            return

        # Show the row that is furthest off; the first bad row is often one
        # where everything is near zero, which makes for a useless example.
        error = np.abs(got - expected)
        error[~np.isfinite(error)] = np.inf
        i = int(np.argmax(error.max(axis=rest)))
        lines = [f'    {bad.sum()} of {len(bad)} rows disagree; '
                 f'row {i} is the furthest off.',
                 '  input:',
                 # Per-row inputs are (n_rows, 2); the rest are constants.
                 block([(label, value[i] if np.ndim(value) == 2 else value)
                        for label, value in inputs]),
                 '  expected:',
                 block(list(zip(labels, expected[i]))),
                 '  you gave:',
                 block(list(zip(labels, got[i])))]
        self.report_fail(name, '\n'.join(lines))

    def run(self, name, fn):
        """Call ``fn``; turn any exception from student code into a failure."""
        try:
            fn()
        except Exception:
            self.report_fail(name, '    your code raised an exception:\n' + '\n'.join(
                '    ' + line for line in traceback.format_exc().strip().splitlines()))

    def summary(self):
        print()
        print(f'{self.passed} passed, {self.failed} failed, {self.skipped} skipped')
        if self.failed == 0 and self.skipped == 0:
            print('Everything matches.  Run create_submission.py to package your handin.')
        return self.failed == 0 and self.skipped == 0


# ---------------------------------------------------------------------------

def check_ex1(checker, theta, theta_dot, sol):
    try:
        from forward_kinematics_RR import forward_kinematics_RR
        from jacobian_link_ends_RR import jacobian_link_ends_RR
        from jacobian_coms_RR import jacobian_coms_RR
    except Exception:
        checker.report_skip('ex1', 'could not import your ex1 files:\n'
                            + traceback.format_exc().strip())
        return

    def fk_positions():
        got = []
        for t in theta:
            f = forward_kinematics_RR(t)
            p1, p2, p3 = (f['H_1_0'][:2, 2], f['H_2_0'][:2, 2], f['H_3_0'][:2, 2])
            got.append([(p1 + p2) / 2, p2, (p2 + p3) / 2, p3])
        checker.compare(
            'forward_kinematics_RR (positions of the four tracked points)',
            POINTS, got,
            np.stack([np.column_stack([sol[f'{p}_x'], sol[f'{p}_y']])
                      for p in ('com1', 'end1', 'com2', 'end2')], axis=1),
            [('theta', theta)])

    def velocities(name, jac_fn, keys, columns):
        got = []
        for t, td in zip(theta, theta_dot):
            J = jac_fn(t)
            got.append([np.asarray(J[k], dtype=float) @ td for k in keys])
        checker.compare(
            name, [f'd/dt {c}' for c in columns], got,
            np.stack([np.column_stack([sol[f'{c}_x_dot'], sol[f'{c}_y_dot']])
                      for c in columns], axis=1),
            [('theta', theta), ('theta_dot', theta_dot)])

    checker.run('forward_kinematics_RR', fk_positions)
    checker.run('jacobian_link_ends_RR', lambda: velocities(
        'jacobian_link_ends_RR', jacobian_link_ends_RR,
        ['J_END_1', 'J_END_2'], ['end1', 'end2']))
    checker.run('jacobian_coms_RR', lambda: velocities(
        'jacobian_coms_RR', jacobian_coms_RR,
        ['J_COM_1', 'J_COM_2'], ['com1', 'com2']))


def check_ex2(checker, theta, sol):
    try:
        from get_joint_torques import get_joint_torques
        from get_grav_comp_torques import get_grav_comp_torques
    except Exception:
        checker.report_skip('ex2', 'could not import your ex2 files:\n'
                            + traceback.format_exc().strip())
        return

    def torques(name, fn, second_arg, arg_label, columns):
        got = [np.asarray(fn(t, second_arg), dtype=float).reshape(-1) for t in theta]
        checker.compare(name, ['tau'], np.asarray(got)[:, None, :],
                        np.column_stack([sol[c] for c in columns])[:, None, :],
                        [('theta', theta), (arg_label, second_arg)])

    checker.run('get_joint_torques', lambda: torques(
        'get_joint_torques', get_joint_torques, DESIRED_FORCE, 'desired_force',
        ['tau_force_1', 'tau_force_2']))
    checker.run('get_grav_comp_torques', lambda: torques(
        'get_grav_comp_torques', get_grav_comp_torques, GRAVITY, 'gravity',
        ['tau_grav_comp_1', 'tau_grav_comp_2']))


def check_pickles(checker, theta_full, sol):
    sl = slice(2400, 3200)      # the slice sample_path.py stores

    path = os.path.join(ROOT, 'ex1', 'results.pkl')
    if not os.path.exists(path):
        checker.report_skip('ex1/results.pkl', 'run sample_path.py from inside ex1/')
    else:
        def run():
            with open(path, 'rb') as f:
                results = pickle.load(f)
            labels, got, expected = [], [], []
            for point, prefix in zip(POINTS, ('com1', 'end1', 'com2', 'end2')):
                for field in ('x', 'y', 'x_dot', 'y_dot'):
                    labels.append(f'{point} {field}')
                    got.append(np.asarray(results[point][field], dtype=float).squeeze())
                    expected.append(sol[f'{prefix}_{field}'][sl])
            checker.compare('ex1/results.pkl', labels,
                            np.stack(got, axis=1), np.stack(expected, axis=1),
                            [('theta', theta_full[sl])])
        checker.run('ex1/results.pkl', run)

    for filename, columns in (('tau_desired_force.pkl', ('tau_force_1', 'tau_force_2')),
                              ('tau_grav_comp.pkl', ('tau_grav_comp_1', 'tau_grav_comp_2'))):
        path = os.path.join(ROOT, 'ex2', filename)
        if not os.path.exists(path):
            checker.report_skip(f'ex2/{filename}',
                                'run sample_torques.py from inside ex2/')
            continue

        def run(path=path, filename=filename, columns=columns):
            with open(path, 'rb') as f:
                got = np.asarray(pickle.load(f), dtype=float)
            checker.compare(f'ex2/{filename}', ['tau'], got[:, None, :],
                            np.column_stack([sol[c] for c in columns])[:, None, :],
                            [('theta', theta_full)])
        checker.run(f'ex2/{filename}', run)


def check_figures(checker):
    for relative in ('ex1/path.png', 'ex2/torques_ee.png', 'ex2/torques_grav_comp.png'):
        if os.path.exists(os.path.join(ROOT, *relative.split('/'))):
            checker.report_pass(relative)
        else:
            checker.report_skip(relative, 'required deliverable, not found')


def main():
    os.environ.setdefault('MPLBACKEND', 'Agg')

    try:
        from common.robot_info import load_data, load_solution
    except Exception:
        print('Could not import common/robot_info.py. Run this from the assignment '
              'root directory:\n    python local_check.py')
        raise

    data = load_data()
    sol_full = load_solution()
    theta = data['theta'][::STRIDE]
    theta_dot = data['theta_dot'][::STRIDE]
    sol = {k: v[::STRIDE] for k, v in sol_full.items()}

    checker = Checker()
    check_ex1(checker, theta, theta_dot, sol)
    check_ex2(checker, theta, sol)
    check_pickles(checker, data['theta'], sol_full)
    check_figures(checker)
    return 0 if checker.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
