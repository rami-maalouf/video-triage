"""Command-line entry point: scan a folder, transcribe, classify, report."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import classify as classify_mod
from . import discover as discover_mod
from . import transcribe as transcribe_mod
from .cache import Cache
from .report import write_report
from .transcribe import DEFAULT_MODEL


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-triage",
        description="Transcribe a folder of videos and route each to a channel.",
    )
    p.add_argument("folder", type=Path, help="Folder of videos to scan (recursive).")
    p.add_argument(
        "--channels",
        type=Path,
        default=Path("channels.yaml"),
        help="YAML file describing your channels (default: channels.yaml).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("report.md"),
        help="Markdown report output path (default: report.md).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path("video_triage.db"),
        help="SQLite cache path (default: video_triage.db).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX Whisper model (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Skip clips shorter than this many seconds (default: 2).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many new videos (for testing).",
    )
    p.add_argument(
        "--reclassify",
        action="store_true",
        help="Re-run classification on already-transcribed videos.",
    )
    p.add_argument(
        "--no-classify",
        action="store_true",
        help="Transcribe only; skip the Claude classification step.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.folder.is_dir():
        log(f"error: {args.folder} is not a directory")
        return 1

    # Channels + Claude client are optional — without them we still transcribe.
    channels = None
    client = None
    if not args.no_classify:
        if not args.channels.is_file():
            log(f"warning: {args.channels} not found — skipping classification.")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            log("warning: ANTHROPIC_API_KEY not set — skipping classification.")
        else:
            import anthropic

            channels = classify_mod.load_channels(args.channels)
            client = anthropic.Anthropic()
            log(f"Loaded {len(channels)} channels; classifying with "
                f"{classify_mod.MODEL}.")

    cache = Cache(args.db)
    videos = discover_mod.discover(args.folder)
    log(f"Found {len(videos)} videos under {args.folder}.")

    processed = 0
    for video in videos:
        cached = cache.get(video.id)
        already_transcribed = cached is not None and not cached["error"]
        needs_classify = (
            client is not None
            and (cached is None or not cached["classified"] or args.reclassify)
        )

        if already_transcribed and not needs_classify:
            continue
        if args.limit is not None and processed >= args.limit and not (
            already_transcribed and needs_classify
        ):
            continue

        try:
            row = _process_one(
                video, cached, channels, client, client and needs_classify, args
            )
            cache.upsert(row)
            if not (already_transcribed):
                processed += 1
        except Exception as exc:  # noqa: BLE001 — record, keep going
            log(f"  error on {video.path.name}: {exc}")
            cache.upsert(
                {
                    "id": video.id,
                    "path": str(video.path),
                    "size": video.size,
                    "mtime": video.mtime,
                    "error": str(exc),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    rows = cache.all()
    write_report(rows, args.out, args.folder)
    cache.close()
    log(f"Wrote report to {args.out} ({len(rows)} videos).")
    return 0


def _process_one(video, cached, channels, client, do_classify, args) -> dict:
    """Transcribe (unless cached) and optionally classify a single video."""
    discover_mod.probe(video)

    transcript = cached["transcript"] if cached else None
    language = cached["language"] if cached else None
    is_cached_transcript = cached is not None and not cached["error"]

    # Heuristic skip: no audio or too short → not worth transcribing.
    too_short = video.duration is not None and video.duration < args.min_duration
    skip_transcribe = (not video.has_audio) or too_short

    if not is_cached_transcript:
        if skip_transcribe:
            transcript = ""
            log(f"  {video.path.name}: skipped transcription "
                f"({'no audio' if not video.has_audio else 'too short'})")
        else:
            log(f"  {video.path.name}: transcribing…")
            result = transcribe_mod.transcribe(video.path, model=args.model)
            transcript = result["text"]
            language = result["language"]

    row = {
        "id": video.id,
        "path": str(video.path),
        "size": video.size,
        "mtime": video.mtime,
        "duration": video.duration,
        "width": video.width,
        "height": video.height,
        "created_at": video.created_at,
        "has_audio": int(video.has_audio),
        "language": language,
        "transcript": transcript,
        "error": None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    if do_classify and client is not None and channels is not None:
        meta = {
            "duration_seconds": round(video.duration) if video.duration else None,
            "orientation": _orientation(video),
            "captured_at": video.created_at,
            "has_audio": video.has_audio,
        }
        log(f"  {video.path.name}: classifying…")
        c = classify_mod.classify(client, transcript or "", meta, channels)
        row.update(
            {
                "is_content": c.is_content,
                "channel": c.channel,
                "confidence": c.confidence,
                "reason": c.reason,
                "suggested_title": c.suggested_title,
                "classified": 1,
            }
        )
    elif cached is not None:
        # Preserve any prior classification when only re-reporting.
        row.update(
            {
                "is_content": cached["is_content"],
                "channel": cached["channel"],
                "confidence": cached["confidence"],
                "reason": cached["reason"],
                "suggested_title": cached["suggested_title"],
                "classified": cached["classified"],
            }
        )

    return row


def _orientation(video) -> str | None:
    v = video.is_vertical
    if v is None:
        return None
    return "vertical" if v else "horizontal"


if __name__ == "__main__":
    raise SystemExit(main())
