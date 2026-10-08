"""Read tracks, cues, and phrase data from the rekordbox database."""

from __future__ import annotations

import logging
from typing import Any

from pyrekordbox import Rekordbox6Database

from djcues.constants import resolve_phrase_label
from djcues.models import BeatGrid, CuePoint, Phrase, Track, WaveformPoint

logger = logging.getLogger(__name__)

_db: Rekordbox6Database | None = None


def get_db() -> Rekordbox6Database:
    """Get or create the shared database connection."""
    global _db
    if _db is None:
        _db = Rekordbox6Database()
    return _db


def find_playlist(name: str) -> Any | None:
    """Find a playlist by exact name. Returns the pyrekordbox playlist object or None.

    Prefers actual playlists over folders when names collide.
    Attribute: 0 or -128 = playlist, 1 = folder.
    """
    db = get_db()
    folder_match = None
    for pl in db.get_playlist():
        if pl.Name == name:
            if pl.Attribute != 1:
                return pl  # actual playlist — return immediately
            folder_match = pl  # folder — keep as fallback
    return folder_match


def _extract_beat_grid(track_content: Any) -> BeatGrid:
    """Extract BPM and first beat position from a track's analysis files."""
    db = get_db()
    bpm = track_content.BPM / 100

    # Try to get first beat from ANLZ beat grid
    first_beat_ms = 0.0
    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".DAT":
                for tag in af.tags:
                    if type(tag).__name__ == "PQTZAnlzTag":
                        times = tag.get_times()
                        if len(times) > 0:
                            # times are in seconds, convert to ms
                            first_beat_ms = float(times[0]) * 1000
                        break
    except Exception as e:
        logger.warning("Could not read beat grid for %s: %s", track_content.Title, e)

    return BeatGrid(first_beat_ms=first_beat_ms, bpm=bpm)


def _extract_phrases(track_content: Any, beat_grid: BeatGrid) -> list[Phrase]:
    """Extract PSSI phrase structure from a track's ANLZ EXT file."""
    db = get_db()
    phrases: list[Phrase] = []

    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".EXT":
                for tag in af.tags:
                    if type(tag).__name__ == "PSSIAnlzTag":
                        content = tag.content
                        mood = content.mood
                        entries = list(content.entries)

                        for i, entry in enumerate(entries):
                            next_beat = (
                                entries[i + 1].beat
                                if i + 1 < len(entries)
                                else content.end_beat
                            )
                            label = resolve_phrase_label(mood, entry.kind)
                            pos_ms = beat_grid.beat_to_ms(entry.beat)
                            end_ms = beat_grid.beat_to_ms(next_beat)

                            phrases.append(Phrase(
                                beat_start=entry.beat,
                                beat_end=next_beat,
                                kind=entry.kind,
                                label=label,
                                position_ms=pos_ms,
                                duration_ms=end_ms - pos_ms,
                            ))
                        break  # only process first PSSI tag
    except Exception as e:
        logger.warning("Could not read phrases for %s: %s", track_content.Title, e)

    return phrases


def _extract_cues(track_content: Any) -> list[CuePoint]:
    """Extract cue points from the database."""
    db = get_db()
    cues: list[CuePoint] = []

    for c in db.get_cue(ContentID=track_content.ID):
        loop_end = None
        if c.OutMsec is not None and c.OutMsec > 0:
            loop_end = float(c.OutMsec)

        cues.append(CuePoint(
            kind=c.Kind,
            position_ms=float(c.InMsec),
            loop_end_ms=loop_end,
            color_table_index=c.ColorTableIndex,
            color=c.Color if c.Color is not None else -1,
            comment=c.Comment or "",
        ))

    return cues


def _extract_waveform(track_content: Any, max_points: int = 800) -> list[WaveformPoint] | None:
    """Extract waveforms from Rekordbox 6/7 ANLZ files.

    The complete pyrekordbox Construct parser can fail on one newer or
    malformed tag and then hide all other tags. We therefore scan the raw
    ANLZ framing and decode PWV3/PWV5 (EXT) plus PWV6/PWV7 (2EX) ourselves.
    """
    db = get_db()
    try:
        # This only resolves paths; unlike read_anlz_files it does not parse
        # every tag and survives a bad PWV5 block.
        paths = db.get_anlz_paths(track_content)
        for key in ("EXT", "2EX", "DAT"):
            path = paths.get(key)
            if path is None or not path.exists():
                continue
            points = _decode_waveform_file(path.read_bytes(), max_points)
            if points:
                return points
    except Exception as e:
        logger.warning("Could not read waveform for %s: %s", track_content.Title, e)
    return None


def _decode_waveform_file(data: bytes, max_points: int = 800) -> list[WaveformPoint]:
    """Decode waveform tags from raw ANLZ bytes without Construct parsing."""
    import struct

    points: list[WaveformPoint] = []
    pos = 0
    if data[:4] == b"ANLZ" and len(data) >= 8:
        file_header_len = struct.unpack_from(">I", data, 4)[0]
        if 12 <= file_header_len <= len(data):
            pos = file_header_len
    while pos + 12 <= len(data):
        tag_type = data[pos : pos + 4]
        if not tag_type.isascii():
            pos += 1
            continue
        header_len, tag_len = struct.unpack_from(">II", data, pos + 4)
        if header_len < 12 or tag_len < header_len or pos + tag_len > len(data):
            # Recover from an invalid block and search for a later valid tag.
            pos += 1
            continue
        if tag_type in (b"PWV3", b"PWV4", b"PWV5", b"PWV6", b"PWV7"):
            points.extend(_decode_waveform_tag(tag_type.decode("ascii"), header_len, data[pos : pos + tag_len]))
        pos += tag_len
    return _downsample_waveform(points, max_points)


def _decode_waveform_tag(tag_type: str, header_len: int, tag_data: bytes) -> list[WaveformPoint]:
    """Decode one PWV tag, tolerating Rekordbox 6/7 header variants."""
    import struct

    if len(tag_data) < header_len:
        return []
    body = tag_data[header_len:]

    if tag_type == "PWV5":
        # Two-byte packed RGB/height entries. Trust payload size over a stale
        # entry count found in some Rekordbox-generated files.
        if len(body) >= 12:
            declared = struct.unpack_from(">I", body, 0)[0]
            usable_declared = declared * 2
            body = body[12 : 12 + usable_declared] if usable_declared <= len(body) - 12 else body[12:]
        usable = len(body) - (len(body) % 2)
        return [
            WaveformPoint(
                height=((value >> 2) & 0x1F) / 31,
                red=(value >> 12) & 0x07,
                green=(value >> 9) & 0x07,
                blue=(value >> 6) & 0x07,
            )
            for (value,) in struct.iter_unpack(">H", body[:usable])
        ]

    if tag_type == "PWV3":
        body = body[12:] if len(body) >= 12 else b""
        return [WaveformPoint(height=(value & 0x1F) / 31, red=1, green=2, blue=5) for value in body]

    if tag_type == "PWV6":
        body = body[8:] if len(body) >= 8 else b""
        usable = len(body) - (len(body) % 3)
        return [
            WaveformPoint(
                height=max(mid, high, low) / 255,
                red=round(low / 255 * 7),
                green=round(mid / 255 * 7),
                blue=round(high / 255 * 7),
            )
            for mid, high, low in zip(body[:usable:3], body[1:usable:3], body[2:usable:3])
        ]

    if tag_type == "PWV7":
        body = body[12:] if len(body) >= 12 else b""
        usable = len(body) - (len(body) % 3)
        return [
            WaveformPoint(
                height=max(mid, high, low) / 255,
                red=round(low / 255 * 7),
                green=round(mid / 255 * 7),
                blue=round(high / 255 * 7),
            )
            for mid, high, low in zip(body[:usable:3], body[1:usable:3], body[2:usable:3])
        ]

    if tag_type == "PWV4":
        body = body[12:] if len(body) >= 12 else b""
        usable = len(body) - (len(body) % 6)
        result: list[WaveformPoint] = []
        for i in range(0, usable, 6):
            _, d2, d3, d4, d5 = body[i + 1 : i + 6]
            result.append(WaveformPoint(
                height=max(d2, d3, d4, d5 & 0x7F) / 127,
                red=round((d3 & 0x7F) / 127 * 7),
                green=round((d4 & 0x7F) / 127 * 7),
                blue=round((d5 & 0x7F) / 127 * 7),
            ))
        return result
    return []


def _downsample_waveform(points: list[WaveformPoint], max_points: int) -> list[WaveformPoint]:
    """Keep peak points while limiting inline SVG size."""
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [
        max(points[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)], key=lambda point: point.height)
        for i in range(max_points)
    ]


def _extract_vocal_track(track_content: Any) -> list[int] | None:
    """Extract PVDI vocal detection data from the .2EX ANLZ file.

    Returns a list of per-frame vocal confidence values (0-4),
    where each frame covers 1024/22050 ≈ 46.4ms.
    """
    import struct
    anlz_rel = track_content.AnalysisDataPath
    if not anlz_rel:
        return None
    base = __import__("pathlib").Path.home() / "Library/Pioneer/rekordbox/share"
    ex2_path = base / anlz_rel.lstrip("/").replace("ANLZ0000.DAT", "ANLZ0000.2EX")
    if not ex2_path.exists():
        return None
    try:
        with open(ex2_path, "rb") as f:
            data = f.read()
        pos = data.find(b"PVDI")
        if pos < 0:
            return None
        header_len = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        tag_len = struct.unpack(">I", data[pos + 8 : pos + 12])[0]
        body = data[pos + header_len : pos + tag_len]
        return list(body)
    except Exception as e:
        logger.warning("Could not read vocal track for %s: %s", track_content.Title, e)
        return None


def load_track(track_content: Any) -> Track:
    """Load a single track with all analysis data."""
    beat_grid = _extract_beat_grid(track_content)
    phrases = _extract_phrases(track_content, beat_grid)
    cues = _extract_cues(track_content)
    waveform = _extract_waveform(track_content)
    vocal_track = _extract_vocal_track(track_content)

    artist_name = ""
    if track_content.Artist:
        artist_name = track_content.Artist.Name or ""

    return Track(
        id=track_content.ID,
        title=track_content.Title or "",
        artist=artist_name,
        bpm=track_content.BPM / 100,
        duration_ms=float(track_content.Length or 0) * 1000,
        analysis_path=track_content.AnalysisDataPath or "",
        cues=cues,
        phrases=phrases,
        beat_grid=beat_grid,
        waveform=waveform,
        vocal_track=vocal_track,
    )


def load_playlist_tracks(playlist_id: int) -> list[Track]:
    """Load all tracks from a playlist with full analysis data."""
    db = get_db()
    songs = list(db.get_playlist_songs(PlaylistID=playlist_id))
    tracks: list[Track] = []
    for song in songs:
        try:
            tracks.append(load_track(song.Content))
        except Exception as e:
            logger.warning("Skipping track %s: %s", song.Content.Title, e)
    return tracks
