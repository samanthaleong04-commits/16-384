# Assignment 3 — Forward Kinematics

Implement the forward kinematics of a planar revolute (RR…R) arm in
`code/Robot.py`. Given the link lengths and the joint angles, `Robot.fk` returns
the chain of homogeneous frames, and the end effector is the last frame. See
`assignment3.pdf` for the full writeup and the written questions.

## Folder layout

- `assignment3.tex`, `assignment3.pdf` — the writeup (source and compiled).
- `code/` — the code you work in:
  - `Robot.py` — **the file you edit and submit.** Fill in `forward_kinematics`.
  - `sample_path.py` — plots your end-effector path against the sample ground
    truth, a quick visual sanity check.
  - `create_submission.py` — packages `Robot.py` into the zip to upload.
  - `sample_ground_truth.csv` — the sample 2-link log `sample_path.py` uses.
- `local_autograder/` — the local self-check (see below).
- `lab/` — the hands-on lab code for the real xArm7 (see below):
  - `fk.py` — **the file you edit.** Fill in `forward_kinematics_RR` and
    `jacobian_RR`.
  - `record.py` / `replay.py` — hand-guide the arm to record a path, and play a
    recording back.
  - `square.csv` — a sample recording, so you can test your code in simulation.
  - `handin/` — where your replay video goes.
- `latex/` — the document class and figures for the writeup.

The due date is on Canvas.

## Editing the writeup

`assignment3.tex` is the writeup source; `latex/` holds `16384_doc.cls` and the
figures. You can upload this folder to Overleaf and use `assignment3.tex` as the
project's main file.

## Working in the code

From `code/`, visualize your forward kinematics on the sample arm:

```bash
cd code
python sample_path.py
```

You need `numpy` and `matplotlib` for the plot.

## Checking your code before you submit

Run the local autograder on your `Robot.py` to check your forward kinematics
against the provided answers. From the `local_autograder/` folder:

```bash
cd local_autograder
python local_autograder.py
```

It auto-finds `../code/Robot.py`. You can also point it at any `Robot.py`:

```bash
python local_autograder.py path/to/Robot.py
python local_autograder.py path/to/code        # a folder that has Robot.py
```

It runs your `Robot.fk` over each test case, compares the end-effector path
against the shipped outputs within 1 cm, and prints per-chain PASS/FAIL plus an
overall result. You only need `numpy` installed.

## Test cases available to you

`local_autograder/` ships 25 arms — five each for chain lengths 2 through 6
(`expected_2dof_1.csv` … `expected_6dof_5.csv`). Each file has 10 joint-angle
inputs and the reference end-effector outputs (`gt_x`, `gt_y`) for one arm (250
cases in all):

| links | link lengths |
|---|---|
| 2 | `[0.55, 0.4]`, `[0.25, 0.65]`, `[0.4, 0.4]`, `[0.7, 0.2]`, `[0.3, 0.5]` |
| 3 | `[0.3, 0.3, 0.3]`, `[0.5, 0.3, 0.2]`, `[0.4, 0.35, 0.25]`, `[0.6, 0.2, 0.2]`, `[0.25, 0.25, 0.5]` |
| 4 | `[0.3, 0.25, 0.2, 0.15]`, `[0.25, 0.25, 0.25, 0.25]`, `[0.4, 0.3, 0.2, 0.1]`, `[0.35, 0.3, 0.2, 0.15]`, `[0.2, 0.2, 0.3, 0.3]` |
| 5 | `[0.2, 0.2, 0.2, 0.2, 0.2]`, `[0.3, 0.25, 0.2, 0.15, 0.1]`, `[0.25, 0.2, 0.2, 0.15, 0.2]`, `[0.3, 0.3, 0.2, 0.1, 0.1]`, `[0.15, 0.25, 0.25, 0.2, 0.15]` |
| 6 | `[0.2, 0.15, 0.15, 0.15, 0.1, 0.1]`, `[0.2, 0.2, 0.15, 0.15, 0.1, 0.1]`, `[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]`, `[0.25, 0.2, 0.15, 0.1, 0.1, 0.1]`, `[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]` |

The link lengths for each file are also on its first line (`# link_lengths=...`).

Gradescope grades these exact same cases, but perturbs each one a little on
every run (it computes the answers from the reference solution, not from any
file). A correct, general `forward_kinematics` passes both; code hard-coded to
these exact numbers passes here but fails on Gradescope.

## The hands-on lab

`lab/` runs your kinematics on a real (or simulated) xArm7, driven as a planar
RR arm. It needs the course robot library, `xarm7_lib`, which this assignment
needs a recent version of:

```bash
git clone https://github.com/CMU-16384-2026-Fall/16384-robot-lib
cd 16384-robot-lib
git pull                        # if you already had it cloned
conda env update -f environment.yml
conda activate 16384
```

Fill in `forward_kinematics_RR` and `jacobian_RR` in `lab/fk.py`, then check
them against the provided sample recording. With `ROBOT_IP` unset this runs the
MuJoCo simulation, so it needs no robot:

```bash
cd lab
python replay.py square.csv
```

In lab, with `ROBOT_IP` set to your station's controller, the same scripts drive
the real arm — `python record.py my-path.csv` to hand-guide a path, then
`python replay.py my-path.csv` to play it back. Put your replay video and its
recording in `lab/handin/`; `create_submission.py` packages them.

See `assignment3.pdf` for the safety rules and the full lab procedure.

## Submitting

From `code/`, build the upload zip and submit it to Gradescope:

```bash
cd code
python create_submission.py
```

It asks for your Andrew ID and writes `<andrewid>_hw3.zip` containing `Robot.py`,
`lab/fk.py`, and everything in `lab/handin/`. Upload that zip to the HW3
autograder on Gradescope.
