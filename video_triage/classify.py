"""Route a transcript to one of the user's channels with a single Claude call."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

# Haiku 4.5: cheapest/fastest model, right for high-volume classification.
MODEL = "claude-haiku-4-5"


@dataclass
class Channel:
    id: str
    name: str
    description: str


@dataclass
class Classification:
    is_content: float  # 0..1 — probability this was meant to be published
    channel: str  # one of the channel ids, or "none" / "unsure"
    confidence: float  # 0..1 — confidence in the channel pick
    reason: str
    suggested_title: str


def load_channels(path: Path) -> list[Channel]:
    """Load channel profiles from a YAML file (see channels.example.yaml)."""
    data = yaml.safe_load(path.read_text())
    channels = []
    for entry in data.get("channels", []):
        channels.append(
            Channel(
                id=str(entry["id"]),
                name=entry.get("name", entry["id"]),
                description=entry.get("description", ""),
            )
        )
    if not channels:
        raise ValueError(f"No channels found in {path}")
    return channels


def _schema(channels: list[Channel]) -> dict:
    channel_ids = [c.id for c in channels] + ["none", "unsure"]
    return {
        "type": "object",
        "properties": {
            "is_content": {
                "type": "number",
                "description": (
                    "Probability from 0 to 1 that this clip was meant to be "
                    "published as content (vs. a personal clip, test shot, "
                    "accidental recording, or raw b-roll)."
                ),
            },
            "channel": {
                "type": "string",
                "enum": channel_ids,
                "description": (
                    "Which channel this clip best fits. Use 'none' if it is "
                    "clearly not content, 'unsure' if it is content but the "
                    "channel is ambiguous."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence from 0 to 1 in the channel choice.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the channel choice.",
            },
            "suggested_title": {
                "type": "string",
                "description": "A short suggested title/caption, or empty string.",
            },
        },
        "required": [
            "is_content",
            "channel",
            "confidence",
            "reason",
            "suggested_title",
        ],
        "additionalProperties": False,
    }


def _prompt(transcript: str, meta: dict, channels: list[Channel]) -> str:
    channel_block = "\n".join(
        f"- {c.id} ({c.name}): {c.description}" for c in channels
    )
    meta_block = "\n".join(f"- {k}: {v}" for k, v in meta.items() if v is not None)
    transcript_block = transcript if transcript else "(no speech detected)"
    return (
        "You are sorting a creator's raw video files. They film a lot and lose "
        "track of which clips were meant to be published and on which account.\n\n"
        "Their channels:\n"
        f"{channel_block}\n\n"
        "Video metadata:\n"
        f"{meta_block}\n\n"
        "Transcript of the video's audio:\n"
        f'"""\n{transcript_block}\n"""\n\n'
        "Decide whether this was meant to be published content and, if so, which "
        "channel it fits best. Vertical short clips with direct-to-camera or "
        "in-public talking are usually content. Base the channel choice on the "
        "transcript topic and tone matching a channel's description."
    )


def classify(
    client,
    transcript: str,
    meta: dict,
    channels: list[Channel],
) -> Classification:
    """Single structured Claude call returning a Classification."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        output_config={"format": {"type": "json_schema", "schema": _schema(channels)}},
        messages=[{"role": "user", "content": _prompt(transcript, meta, channels)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    data = json.loads(text)
    return Classification(
        is_content=float(data.get("is_content", 0.0)),
        channel=str(data.get("channel", "unsure")),
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", "")),
        suggested_title=str(data.get("suggested_title", "")),
    )
