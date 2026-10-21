# video-triage

Point it at a folder of videos. It transcribes each one locally (MLX Whisper on
Apple Silicon — free, private, fast), then asks Claude which of your channels the
clip belongs to and how likely it was meant to be published. Output is a Markdown
report. Re-runs are incremental: a SQLite cache means it never reprocesses a clip
it has already seen.

No image classification — routing is based on the transcript plus basic metadata
(duration, orientation, capture date).

## Setup

Requires `ffmpeg` (transcription) and an `ANTHROPIC_API_KEY` (classification only).

```sh
brew install ffmpeg          # if you don't have it
cp channels.example.yaml channels.yaml   # then edit for your accounts
export ANTHROPIC_API_KEY=sk-ant-...      # classification step only
```

## Getting videos off your iPhone

Plug the phone in via USB and use the built-in **Image Capture** app (or Finder /
Photos import): select all, import to one folder. Faster and more reliable than
AirDrop for bulk, and it preserves capture dates. An external SSD is only needed
if the library is larger than your free disk space — a local folder is fine
otherwise.

## Usage

```sh
# First run: test on a handful before committing to the whole library
uv run video-triage ~/Videos --channels channels.yaml --limit 5

# Full run
uv run video-triage ~/Videos --channels channels.yaml --out report.md
```

Useful flags:

- `--model mlx-community/whisper-small` — faster, rougher transcription
- `--no-classify` — transcribe only (no API key needed)
- `--reclassify` — re-run channel routing after editing `channels.yaml`
- `--min-duration 2` — skip clips shorter than N seconds
- `--db video_triage.db` — where the incremental cache lives

The first run downloads the Whisper model (~1.5 GB for the default turbo model)
once; later runs reuse it.

## How it decides

1. **Metadata filter** — clips with no audio track or under `--min-duration` are
   skipped before transcription.
2. **Transcribe** — `ffmpeg` extracts 16 kHz mono audio; MLX Whisper transcribes it.
3. **Classify** — one Claude (Haiku) call per clip with your channel profiles +
   transcript + metadata returns `is_content` (0–1), the best `channel`, a
   confidence, a one-line reason, and a suggested title.
4. **Report** — `report.md`, sorted by content likelihood, with a per-channel
   summary and transcripts for likely-content clips. Everything is also in the
   SQLite DB for later (e.g. a future Notion sync).
