"""Package your work for Gradescope.

    python create_submission.py

Runs local_check.py first so you know whether anything is obviously wrong, then
writes <andrew_id>.zip containing your ex1/ and ex2/ folders.  Upload that zip
to the "Assignment 2 (Code)" Gradescope assignment.
"""

import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ['ex1', 'ex2']
SKIP_DIRS = {'__pycache__', '.git', '.ipynb_checkpoints'}
SKIP_FILES = {'.DS_Store'}


def run_self_check():
    """Return True if autograder.py reported no problems."""
    sys.path.insert(0, ROOT)
    try:
        import autograder
    except Exception as exc:
        print(f'Could not run autograder.py ({exc}); packaging anyway.')
        return True
    return autograder.main() == 0


def collect():
    """Every file we are going to hand in, as (path on disk, path in zip)."""
    entries = []
    for folder in FOLDERS:
        base = os.path.join(ROOT, folder)
        if not os.path.isdir(base):
            raise SystemExit(f'Could not find the {folder}/ folder next to this script.')
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name in SKIP_FILES:
                    continue
                path = os.path.join(root, name)
                entries.append((path, os.path.relpath(path, ROOT)))
    return entries


def create_submission():
    ok = run_self_check()
    if not ok:
        print()
        answer = input('The self-check found problems. Package anyway? [y/N] ').strip()
        if answer.lower() not in ('y', 'yes'):
            print('Nothing written.')
            return 1

    entries = collect()
    output = os.path.join(ROOT, f'submission.zip')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname)

    print()
    print(f'Wrote {output} with {len(entries)} files:')
    for _, arcname in entries:
        print(f'  {arcname}')
    print()
    print('Rename this file to <andrew_id>.zip and upload it to Gradescope.')
    return 0


if __name__ == '__main__':
    sys.exit(create_submission())
