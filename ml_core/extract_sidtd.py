"""Safely extract the SIDTD benchmark archive on Windows or Linux."""

import argparse
from pathlib import Path
import zipfile


def extract_archive(archive: Path, output_dir: Path) -> None:
    """Extract a SIDTD zip while rejecting path-traversal entries."""

    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir.resolve()

    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zip_file.extractall(destination)

    print(f"Extracted {len(zip_file.infolist())} archive entries into {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True, help="Path to templates.zip")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    extract_archive(args.archive, args.output)
