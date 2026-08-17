#!/usr/bin/env python3
"""Fetch and unpack the speechocean762 corpus.

speechocean762 is CC BY 4.0: 5000 English utterances read by Mandarin-native
learners, scored independently by five experts at the utterance, word and
phoneme level. See training/README.md for the attribution this obliges.

OpenSLR does not publish a checksum for the archive, so the first successful
download records one in corpus.lock.json and every later run is verified
against it. Pass --expect-sha256 if you have a hash from a source you trust
more than your own first download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URL = "https://www.openslr.org/resources/101/speechocean762.tar.gz"
LOCK_PATH = HERE / "corpus.lock.json"
CHUNK = 1 << 20

# Unpacking is only worth attempting if the archive really is the corpus.
# The corpus README draws scores.json at the top level; the archive puts it
# under resource/. The archive is what ships, so it is what we check.
SCORES_REL = "resource/scores.json"
REQUIRED_ENTRIES = (
    SCORES_REL,
    "train/text",
    "train/utt2spk",
    "train/spk2age",
    "test/text",
    "test/utt2spk",
    "WAVE",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as res:  # noqa: S310 - operator-supplied URL
        total = int(res.headers.get("Content-Length") or 0)
        seen = 0
        last = -1
        with tmp.open("wb") as out:
            while True:
                block = res.read(CHUNK)
                if not block:
                    break
                out.write(block)
                seen += len(block)
                if total:
                    pct = seen * 100 // total
                    if pct != last and pct % 5 == 0:
                        last = pct
                        print(f"  {pct:3d}%  {seen / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    tmp.replace(dest)


def load_lock() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {}


def resolve_root(extract_dir: Path) -> Path:
    """The archive may or may not carry a single top-level directory."""
    if (extract_dir / SCORES_REL).exists():
        return extract_dir
    for child in sorted(p for p in extract_dir.iterdir() if p.is_dir()):
        if (child / SCORES_REL).exists():
            return child
    raise SystemExit(f"{SCORES_REL} not found under {extract_dir}")


def safe_extract(archive: Path, dest: Path) -> None:
    """Refuse members that would escape dest. Python <3.12 extracts them happily."""
    dest = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest):
                raise SystemExit(f"archive member escapes destination: {member.name}")
            if member.issym() or member.islnk():
                raise SystemExit(f"archive contains a link: {member.name}")
        tar.extractall(dest)  # noqa: S202 - members checked above


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "speechocean762.tar.gz"
    corpus = args.out / "speechocean762"

    if corpus.exists() and (corpus / SCORES_REL).exists():
        print(f"corpus already unpacked at {corpus}")
        return 0

    lock = load_lock()
    expected = args.expect_sha256 or lock.get("sha256")

    if not archive.exists():
        download(args.url, archive)

    digest = sha256_of(archive)
    if expected and digest != expected:
        archive.unlink()
        print(
            f"SHA256 mismatch\n  expected {expected}\n  actual   {digest}\n"
            "The archive was removed. Re-run to download again.",
            file=sys.stderr,
        )
        return 1

    staging = args.out / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    print("unpacking")
    safe_extract(archive, staging)

    root = resolve_root(staging)
    missing = [e for e in REQUIRED_ENTRIES if not (root / e).exists()]
    if missing:
        print(f"archive is missing expected entries: {missing}", file=sys.stderr)
        return 1

    if corpus.exists():
        shutil.rmtree(corpus)
    root.rename(corpus)
    shutil.rmtree(staging, ignore_errors=True)

    if not expected:
        LOCK_PATH.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "sha256": digest,
                    "bytes": archive.stat().st_size,
                    "note": "recorded on first download; not published upstream",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"recorded sha256 {digest} -> {LOCK_PATH.name}")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print(f"corpus ready at {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
