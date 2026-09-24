import os
import zipfile

# The hands-on lab deliverable: everything the student drops in lab/handin
# (their replay video and the recording it replays) goes in the zip too.
LAB_HANDIN = os.path.join('..', 'lab', 'handin')

def lab_handin_files():
    """(path, name-in-zip) for every file in lab/handin, deepest paths last."""
    found = []
    for directory, _subdirs, filenames in os.walk(LAB_HANDIN):
        for filename in sorted(filenames):
            if filename == 'README.md':  # the folder's own instructions
                continue
            path = os.path.join(directory, filename)
            name = os.path.join('lab', 'handin',
                                os.path.relpath(path, LAB_HANDIN))
            found.append((path, name))
    return found

def create_submission():
    # List of (file, name it gets in the zip) to include in the submission
    files_to_include = [
        ('Robot.py', 'Robot.py'),
        (os.path.join('..', 'lab', 'fk.py'), os.path.join('lab', 'fk.py')),
    ]

    handin = lab_handin_files()
    if not handin:
        print(f"Warning: no files in {LAB_HANDIN} -- the hands-on lab video is "
              "not in this submission.")
    files_to_include.extend(handin)

    # Prompt for Andrew ID and name the zip file accordingly
    andrew_id = input("Enter your andrew ID: ").strip()
    zip_filename = f"{andrew_id}_hw3.zip"

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file, name in files_to_include:
            if os.path.exists(file):
                zipf.write(file, name)
                print(f"Added {name} to the submission zip.")
            else:
                print(f"Warning: {file} not found and not included in the submission.")

    print(f"\nSubmission zip created: {zip_filename}")
    print("Please upload this file to Gradescope.")

if __name__ == "__main__":
    create_submission()