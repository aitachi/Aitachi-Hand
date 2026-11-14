"""Create the GitHub repo (if needed) and push the project via the Git Data
REST API (proxy-safe: many small JSON posts instead of one large pack).

Token resolution order: env GITHUB_TOKEN -> file AHAND_TOKEN_FILE.
NEVER commit the token. Run from the repository root.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "aitachi/Aitachi-Hand"
API_ROOT = "https://api.github.com"

# Files/dirs to publish (relative). Results: summaries only, no checkpoints.
INCLUDE_DIRS = ["envs", "algorithms", "scripts", "textbook", "paper", "figures", "docs", "results"]
INCLUDE_FILES = ["README.md", ".gitignore", "LICENSE", "requirements.txt"]
EXCLUDE_PARTS = {"__pycache__", "checkpoints", "logs", "build1.log", "build2.log"}
EXCLUDE_SUFFIXES = (".aux", ".log", ".out", ".toc", ".pyc", ".zip", ".obj", ".tmp")
EXCLUDE_NAMES = {"final_model.zip"}
MAX_BYTES = 40 * 1024 * 1024


def token() -> str:
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"].strip()
    tf = Path(os.environ.get("AHAND_TOKEN_FILE", str(Path.home() / ".ahand_gh_token")))
    if tf.is_file():
        return tf.read_text(encoding="utf-8").strip()
    sys.exit("no token: set GITHUB_TOKEN or create ~/.ahand_gh_token")


def api(url: str, data=None, method=None, tok: str = ""):
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json",
                 "User-Agent": "aitachi-hand-push"},
        data=json.dumps(data).encode() if data is not None else None,
        method=method or ("POST" if data is not None else "GET"))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print("retry", attempt, e)
            time.sleep(2 * (attempt + 1))
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github+json",
                         "User-Agent": "aitachi-hand-push"},
                data=json.dumps(data).encode() if data is not None else None,
                method=method or ("POST" if data is not None else "GET"))


def collect_files(root: Path):
    files = []
    for name in INCLUDE_FILES:
        p = root / name
        if p.is_file():
            files.append(p)
    for d in INCLUDE_DIRS:
        for p in sorted((root / d).rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part in EXCLUDE_PARTS for part in rel.parts):
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIXES or p.name in EXCLUDE_NAMES:
                continue
            if p.stat().st_size > MAX_BYTES:
                print("skip (too big):", rel)
                continue
            files.append(p)
    return files


def main():
    tok = token()
    root = Path(__file__).resolve().parents[1]

    try:
        api(f"{API_ROOT}/repos/{REPO}", tok=tok)
        print("repo exists")
    except Exception:
        repo = api(f"{API_ROOT}/user/repos", {"name": "Aitachi-Hand", "description":
                   "Open 17-DOF dexterous-hand platform + controlled PPO-improvement RL study",
                   "private": False, "has_issues": True}, tok=tok)
        print("repo created:", repo.get("html_url"))

    files = collect_files(root)
    print(f"publishing {len(files)} files")

    # An empty repo rejects the blobs API with 409 "Git Repository is empty".
    # Bootstrap the first commit via the contents API if needed.
    try:
        api(f"{API_ROOT}/repos/{REPO}/commits?per_page=1", tok=tok)
    except Exception:
        readme = (root / "README.md").read_bytes()
        api(f"{API_ROOT}/repos/{REPO}/contents/README.md",
            {"message": "bootstrap", "content": base64.b64encode(readme).decode(),
             "branch": "main"}, method="PUT", tok=tok)
        print("bootstrap commit created")

    tree_items = []
    for i, path in enumerate(files):
        rel = path.relative_to(root).as_posix()
        b64 = base64.b64encode(path.read_bytes()).decode()
        blob = api(f"{API_ROOT}/repos/{REPO}/git/blobs",
                   {"content": b64, "encoding": "base64"}, tok=tok)
        tree_items.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  [{i + 1}/{len(files)}] {rel}")

    tree = api(f"{API_ROOT}/repos/{REPO}/git/trees",
               {"tree": tree_items}, tok=tok)

    head = None
    try:
        commits = api(f"{API_ROOT}/repos/{REPO}/commits?per_page=1", tok=tok)
        head = commits[0]["sha"]
    except Exception:
        pass

    commit_data = {
        "message": "Aitachi-Hand v1.0: platform, dual-task RL envs, 9-method controlled study, textbook, IEEE paper",
        "tree": tree["sha"],
    }
    if head:
        commit_data["parents"] = [head]
    commit = api(f"{API_ROOT}/repos/{REPO}/git/commits", commit_data, tok=tok)
    ref = api(f"{API_ROOT}/repos/{REPO}/git/refs/heads/main",
              {"sha": commit["sha"], "force": False}, method="PATCH", tok=tok) if head else \
        api(f"{API_ROOT}/repos/{REPO}/git/refs",
            {"ref": "refs/heads/main", "sha": commit["sha"]}, tok=tok)
    print("pushed:", commit["sha"][:7], "->", REPO)


if __name__ == "__main__":
    main()
