"""Find video files and read their metadata with ffprobe."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".hevc", ".3gp"}


@dataclass
class VideoFile:
    path: Path
    id: str
    size: int
    mtime: float
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    created_at: str | None = None  # original capture time, if the container records it
    has_audio: bool = False

    @property
    def is_vertical(self) -> bool | None:
        if not self.width or not self.height:
            return None
        return self.height >= self.width


def stable_id(path: Path, size: int, mtime: float) -> str:
    """A content-stable id: same file at same path/size/mtime → same id.

    Used so re-runs skip videos already processed."""
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(str(size).encode())
    h.update(str(int(mtime)).encode())
    return h.hexdigest()[:16]


def discover(folder: Path) -> list[VideoFile]:
    """Recursively find video files under folder, sorted by path."""
    videos: list[VideoFile] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            st = p.stat()
            videos.append(
                VideoFile(
                    path=p,
                    id=stable_id(p, st.st_size, st.st_mtime),
                    size=st.st_size,
                    mtime=st.st_mtime,
                )
            )
    return videos


def probe(video: VideoFile) -> VideoFile:
    """Fill in duration, dimensions, capture time, and audio presence via ffprobe.

    Mutates and returns the same VideoFile. Failures leave fields as None."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video.path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        data = json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return video

    fmt = data.get("format", {})
    try:
        video.duration = float(fmt.get("duration")) if fmt.get("duration") else None
    except (TypeError, ValueError):
        video.duration = None

    video.created_at = (fmt.get("tags") or {}).get("creation_time")

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video.width is None:
            video.width = stream.get("width")
            video.height = stream.get("height")
            if not video.created_at:
                video.created_at = (stream.get("tags") or {}).get("creation_time")
        elif codec_type == "audio":
            video.has_audio = True

    return video
