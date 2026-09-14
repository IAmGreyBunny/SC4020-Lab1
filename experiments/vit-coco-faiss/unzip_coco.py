"""Extract the COCO unlabeled2017 zip into the team data folder.

New file for this experiment. Extracts with multiple threads and skips
files that were already fully extracted, so it can be re-run to resume.
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

_local = threading.local()


def get_zip_handle(zip_path):
    # One ZipFile handle per worker thread; ZipFile is not thread-safe to share.
    if not hasattr(_local, "handle"):
        _local.handle = ZipFile(zip_path)
    return _local.handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    source = Path(args.zip)
    destination = Path(args.dest)

    with ZipFile(source) as archive:
        members = [
            (item.filename, item.file_size)
            for item in archive.infolist()
            if item.filename.lower().endswith((".jpg", ".jpeg"))
        ]
    members.sort()
    print(f"zip members: {len(members)}", flush=True)

    def extract(member):
        name, size = member
        target = destination / name
        # Resume support: skip only when the existing file is the exact size.
        if target.exists() and target.stat().st_size == size:
            return "skipped"
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = get_zip_handle(source)
        data = handle.read(name)
        target.write_bytes(data)
        return "extracted"

    skipped = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, result in enumerate(pool.map(extract, members), start=1):
            if result == "skipped":
                skipped += 1
            if count % 5000 == 0:
                rate = count / (time.time() - start)
                print(
                    f"{count}/{len(members)} "
                    f"({rate:.0f} files/s, skipped {skipped})",
                    flush=True,
                )

    print(
        f"DONE extracted={len(members) - skipped} skipped={skipped} "
        f"elapsed={(time.time() - start) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
