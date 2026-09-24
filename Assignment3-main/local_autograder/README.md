# Local autograder — Assignment 3 (Forward Kinematics)

Check your `Robot.forward_kinematics` before you submit. This runs your `Robot.fk`
over the published test cases (five arms each for chain lengths 2 through 6, 25
in all) and compares the end-effector path against the provided answers, within
1 cm.

## Files

- `local_autograder.py` — the checker. Loads your `Robot.py`, runs it, compares.
- `expected_*.csv` — the 25 published test cases (`expected_2dof_1.csv` …
  `expected_6dof_5.csv`, five per chain length). Each holds the joint-angle
  inputs and the expected end-effector outputs (`gt_x`, `gt_y`) for one arm; the
  link lengths are in the first line (`# link_lengths=...`). No solution code
  here, just the answers.

## How to run

From this `local_autograder/` folder, with your code in the sibling `code/`
folder (the default in the assignment):

```bash
python local_autograder.py
```

It auto-finds `../code/Robot.py`. You can also point it at any `Robot.py`:

```bash
python local_autograder.py path/to/Robot.py
python local_autograder.py path/to/code        # a folder that has Robot.py
```

You only need `numpy` installed.

## What you'll see

```
Using .../code/Robot.py

2-link (link_lengths=[0.55, 0.4]):
  10/10 poses within 1 cm  PASS
3-link (link_lengths=[0.3, 0.3, 0.3]):
  10/10 poses within 1 cm  PASS
...
PASS -- your forward kinematics matches the reference on every arm.
```

An arm that FAILs prints how many poses were off (e.g. `FAIL -- 4/10 poses off
by > 1 cm`), so the failing chain length tells you where to look in your
homogeneous transforms.

## Note on Gradescope

Gradescope grades the exact same 25 arms, but generates fresh random joint
angles and perturbs the link lengths a little on every run (it computes the
answers from the reference solution, not from any file). A correct, general
`forward_kinematics` passes both; code hard-coded to these exact published
numbers passes here but fails on Gradescope.
