import os
import zipfile

CODE = [
    (os.path.join("ex1", "forward_kinematics_RRR.py"), "forward_kinematics_RRR.py"),
    (os.path.join("ex2", "workspace_analysis_PR.py"), "workspace_analysis_PR.py"),
    (os.path.join("ex2", "workspace_analysis_RRR.py"), "workspace_analysis_RRR.py"),
]

VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".gif", ".m4v")


def ex3_files():
    if not os.path.isdir("ex3"):
        return []
    return sorted(
        name for name in os.listdir("ex3")
        if os.path.isfile(os.path.join("ex3", name)) and name != "readme.txt"
    )


def create_submission():
    missing = [src for src, _ in CODE if not os.path.exists(src)]
    if missing:
        raise SystemExit(
            "Missing " + ", ".join(missing) + " — run this from the assignment root."
        )

    videos = ex3_files()
    with zipfile.ZipFile("handin.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for src, dest in CODE:
            zf.write(src, dest)
        for name in videos:
            zf.write(os.path.join("ex3", name), os.path.join("ex3", name))

    print("Created submission archive: handin.zip")
    for _, dest in CODE:
        print(f"  {dest}")
    for name in videos:
        print(f"  ex3/{name}")

    if not videos:
        print("\nWARNING: ex3/ has no videos in it, and the demo video is worth 20 marks.")
    elif not any(name.lower().endswith(VIDEO_SUFFIXES) for name in videos):
        print("\nWARNING: nothing in ex3/ looks like a video; check the file types.")


if __name__ == "__main__":
    create_submission()
