#!/usr/bin/env python3
"""Convert the speechocean762 corpus into this project's neutral training schema.

Everything downstream reads `utterances.jsonl`, never the corpus layout, so a
second corpus only needs another script shaped like this one.

The corpus README and the shipped archive disagree in three places, and the
archive wins here: `scores.json` lives under `resource/`, sentence-level
`completeness` is on a 0-10 scale rather than 0-1, and a word's `phones` is a
list rather than a space-separated string.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Word-level three-way bins.
#
# `total` is not spread evenly: 87.2% of words are a flat 10 and the rest thin
# out below. Cutting at 10 and at 6 separates a perfect word from an accented
# one from a wrong one, and leaves the two minority classes about the same size
# (6.2% and 6.6%) so a classifier is not chasing a 1% tail. See the histogram in
# prepare_report.json before moving these.
WORD_GOOD_MIN = 10.0
WORD_AVERAGE_MIN = 6.0
WORD_LEVELS = ("good", "average", "bad")

# The corpus mixes school children with adults. Age is not observable at
# inference time, so it can only ever be a filter, never a feature.
ADULT_AGE = 18

SPLITS = ("train", "test")


def read_kaldi(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def word_level(total: float) -> str:
    if total >= WORD_GOOD_MIN:
        return "good"
    if total >= WORD_AVERAGE_MIN:
        return "average"
    return "bad"


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=HERE / "data" / "speechocean762")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "utterances.jsonl")
    ap.add_argument("--report", type=Path, default=HERE / "data" / "prepare_report.json")
    args = ap.parse_args()

    scores = json.loads((args.corpus / "resource" / "scores.json").read_text(encoding="utf-8"))

    rows = []
    missing_scores: list[str] = []
    missing_audio: list[str] = []
    speakers_per_split = {}

    for split in SPLITS:
        base = args.corpus / split
        utt2spk = read_kaldi(base / "utt2spk")
        text = read_kaldi(base / "text")
        wav_scp = read_kaldi(base / "wav.scp")
        spk2age = read_kaldi(base / "spk2age")
        spk2gender = read_kaldi(base / "spk2gender")
        speakers_per_split[split] = set(utt2spk.values())

        for utt_id, speaker_id in sorted(utt2spk.items()):
            entry = scores.get(utt_id)
            if entry is None:
                missing_scores.append(utt_id)
                continue
            rel = wav_scp.get(utt_id)
            audio = args.corpus / rel if rel else None
            if audio is None or not audio.exists():
                missing_audio.append(utt_id)
                continue

            age = int(spk2age.get(speaker_id, 0) or 0)
            words = []
            for w in entry["words"]:
                phones = w.get("phones") or []
                if isinstance(phones, str):
                    phones = phones.split()
                words.append(
                    {
                        "text": w["text"],
                        "accuracy": float(w["accuracy"]),
                        "stress": float(w["stress"]),
                        "total": float(w["total"]),
                        "level": word_level(float(w["total"])),
                        "phones": phones,
                        "phoneAccuracy": [float(x) for x in (w.get("phones-accuracy") or [])],
                    }
                )

            rows.append(
                {
                    "utteranceId": utt_id,
                    "speakerId": speaker_id,
                    "split": split,
                    "text": text.get(utt_id, entry.get("text", "")),
                    "audioPath": rel,
                    "durationSec": round(wav_duration(audio), 3),
                    "speaker": {
                        "age": age,
                        "gender": spk2gender.get(speaker_id, ""),
                        "adult": age >= ADULT_AGE,
                    },
                    "labels": {
                        "accuracy": float(entry["accuracy"]),
                        "completeness": float(entry["completeness"]),
                        "fluency": float(entry["fluency"]),
                        "prosodic": float(entry["prosodic"]),
                        "total": float(entry["total"]),
                    },
                    "words": words,
                }
            )

    overlap = speakers_per_split["train"] & speakers_per_split["test"]
    if overlap:
        raise SystemExit(f"train and test share {len(overlap)} speakers; the split is unusable")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(rows, missing_scores, missing_audio)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} utterances -> {args.out}")
    for split in SPLITS:
        n = sum(1 for r in rows if r["split"] == split)
        adults = sum(1 for r in rows if r["split"] == split and r["speaker"]["adult"])
        print(f"  {split}: {n} utterances, {adults} from adults")
    lv = report["words"]["levels"]
    print(f"  word levels: {lv}")
    if missing_scores or missing_audio:
        print(f"  skipped: {len(missing_scores)} without scores, {len(missing_audio)} without audio")
    return 0


def build_report(rows: list[dict], missing_scores: list[str], missing_audio: list[str]) -> dict:
    def summarize(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 4),
            "median": statistics.median(values),
        }

    word_total = collections.Counter()
    levels = collections.Counter()
    stress = collections.Counter()
    accuracy = collections.Counter()
    for r in rows:
        for w in r["words"]:
            word_total[w["total"]] += 1
            levels[w["level"]] += 1
            stress[w["stress"]] += 1
            accuracy[w["accuracy"]] += 1

    lengths = [len(r["words"]) for r in rows]
    return {
        "utterances": {
            "n": len(rows),
            "bySplit": collections.Counter(r["split"] for r in rows),
            "adults": sum(1 for r in rows if r["speaker"]["adult"]),
            "wordsPerUtterance": summarize([float(x) for x in lengths]),
            "durationSec": summarize([r["durationSec"] for r in rows]),
            "labels": {
                k: summarize([r["labels"][k] for r in rows])
                for k in ("accuracy", "completeness", "fluency", "prosodic", "total")
            },
        },
        "words": {
            "n": sum(word_total.values()),
            "levels": dict(levels),
            "levelBins": {
                "good": f"total >= {WORD_GOOD_MIN}",
                "average": f"{WORD_AVERAGE_MIN} <= total < {WORD_GOOD_MIN}",
                "bad": f"total < {WORD_AVERAGE_MIN}",
            },
            "totalHistogram": {str(k): v for k, v in sorted(word_total.items(), reverse=True)},
            "stressHistogram": {str(k): v for k, v in sorted(stress.items(), reverse=True)},
            "accuracyHistogram": {str(k): v for k, v in sorted(accuracy.items(), reverse=True)},
        },
        "skipped": {"noScores": missing_scores, "noAudio": missing_audio},
    }


if __name__ == "__main__":
    raise SystemExit(main())
