"""Fetch the official ORCA description assets (MuJoCo model + targets).

The 111 MB model is NOT committed to git. This script downloads the official
orcahand_description repository archive and stages the needed v2 scene into
assets/orca_v2 (or set AHAND_MODEL_ROOT to an existing copy, e.g. the local
AHand workspace, and skip this script entirely).
"""
from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
URL_CANDIDATES = [
    "https://codeload.github.com/orcahand/orcahand_description/zip/refs/heads/main",
    "https://github.com/orcahand/orcahand_description/archive/refs/heads/main.zip",
]


def download(url: str, dest: Path):
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "aitachi-hand-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main():
    target = ASSETS / "orca_v2"
    if (target / "scene_right.xml").is_file():
        print("assets already present:", target)
        return 0
    ASSETS.mkdir(parents=True, exist_ok=True)
    archive = ASSETS / "orcahand_description.zip"
    last_error = None
    for url in URL_CANDIDATES:
        try:
            download(url, archive)
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            print("failed:", e)
    else:
        print("all mirrors failed:", last_error)
        return 1
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        prefix = names[0].split("/")[0]
        wanted = [n for n in names if "/v2/" in n and n.endswith((".xml", ".stl", ".obj", ".png"))]
        print(f"extracting {len(wanted)} files from {prefix}")
        for n in wanted:
            rel = Path(n).relative_to(prefix)
            out_path = ASSETS / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zf.read(n))
    scene = ASSETS / "v2" / "scene_right.xml"
    if not scene.is_file():
        print("scene_right.xml missing after extraction; inspect assets/")
        return 1
    print("done. set AHAND_MODEL_ROOT=", (ASSETS / "v2").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
