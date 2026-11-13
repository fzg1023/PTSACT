import csv

from .common import sha256


def verify_manifest(root):
    rows = list(csv.DictReader((root / "MANIFEST_SHA256.csv").open(encoding="utf-8", newline="")))
    for index, row in enumerate(rows, 1):
        path = (root / row["relative_path"]).resolve()
        path.relative_to(root.resolve())
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError("Release integrity failure: " + row["relative_path"])
        if index % 50 == 0 or index == len(rows):
            print("[integrity {}/{}]".format(index, len(rows)), flush=True)
    return dict(verified_files=len(rows))
