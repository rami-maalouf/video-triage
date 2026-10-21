"""Extract audio with ffmpeg and transcribe it locally with MLX Whisper."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

# whisper-large-v3-turbo: best speed/accuracy balance on Apple Silicon.
# Override with --model (e.g. mlx-community/whisper-small for faster/rougher).
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def extract_audio(video_path: Path) -> Path:
    """Decode the audio track to a temp 16 kHz mono WAV (what Whisper wants).

    Returns the temp path; caller is responsible for deleting it."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vn",  # drop video — we only decode the audio stream, so this is fast
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            tmp.name,
        ],
        capture_output=True,
        check=True,
        timeout=600,
    )
    return Path(tmp.name)


def transcribe(video_path: Path, model: str = DEFAULT_MODEL) -> dict:
    """Transcribe a video's audio. Returns {"text", "language"}.

    Importing mlx_whisper is deferred so the rest of the CLI stays fast to start."""
    import mlx_whisper

    audio_path = extract_audio(video_path)
    try:
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model)
    finally:
        audio_path.unlink(missing_ok=True)

    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language"),
    }
