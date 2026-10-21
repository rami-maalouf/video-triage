"""Render the SQLite results into a sorted Markdown report."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _fmt_duration(seconds: float | None) -> str:
    if not seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(value * 100)}%"


def write_report(rows: list[sqlite3.Row], out_path: Path, root: Path) -> None:
    """Write a Markdown report: a summary, a sorted table, then transcripts."""
    lines: list[str] = []
    lines.append("# Video triage report\n")
    lines.append(f"Scanned **{len(rows)}** videos under `{root}`.\n")

    likely = [r for r in rows if (r["is_content"] or 0) >= 0.5]
    lines.append(f"**{len(likely)}** look like publishable content (≥50%).\n")

    # Per-channel counts among likely content
    by_channel: dict[str, int] = {}
    for r in likely:
        ch = r["channel"] or "unsure"
        by_channel[ch] = by_channel.get(ch, 0) + 1
    if by_channel:
        lines.append("**Likely content by channel:**\n")
        for ch, n in sorted(by_channel.items(), key=lambda x: -x[1]):
            lines.append(f"- {ch}: {n}")
        lines.append("")

    lines.append("## All videos (sorted by content likelihood)\n")
    lines.append("| Content | Channel | Conf | Dur | Captured | File | Why |")
    lines.append("|--------:|---------|-----:|-----|----------|------|-----|")
    for r in rows:
        try:
            rel = Path(r["path"]).relative_to(root)
        except ValueError:
            rel = Path(r["path"]).name
        note = r["reason"] or (r["error"] and f"⚠️ {r['error']}") or ""
        lines.append(
            f"| {_fmt_pct(r['is_content'])} "
            f"| {r['channel'] or '—'} "
            f"| {_fmt_pct(r['confidence'])} "
            f"| {_fmt_duration(r['duration'])} "
            f"| {(r['created_at'] or '—')} "
            f"| `{rel}` "
            f"| {note} |"
        )
    lines.append("")

    lines.append("## Transcripts (likely content)\n")
    for r in likely:
        rel = Path(r["path"]).name
        title = r["suggested_title"] or rel
        lines.append(f"### {title}")
        lines.append(f"`{r['path']}`\n")
        if r["transcript"]:
            snippet = r["transcript"][:1000]
            if len(r["transcript"]) > 1000:
                snippet += " …"
            lines.append(f"> {snippet}\n")
        else:
            lines.append("> (no speech detected)\n")

    out_path.write_text("\n".join(lines))
