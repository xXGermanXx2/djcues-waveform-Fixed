import struct

from djcues.db import _decode_phrase_file
from djcues.models import BeatGrid


def _tag(tag_type: bytes, payload: bytes, header_len: int = 32) -> bytes:
    header = struct.pack(">4sII", tag_type, header_len, header_len + len(payload))
    return header + (b"\0" * (header_len - 12)) + payload


def _file(body: bytes) -> bytes:
    header_len = 32
    return struct.pack(">4sIII", b"ANLZ", header_len, header_len + len(body), 0) + (b"\0" * 16) + body


def test_decodes_pssi_phrases_from_raw_anlz():
    content_header = struct.pack(">IHH", 24, 2, 1) + b"\0" * 6 + struct.pack(">H", 65) + b"\0" * 2 + b"\0\0"
    entry_a = struct.pack(">HHH", 0, 1, 1) + b"\0" * 18
    entry_b = struct.pack(">HHH", 1, 33, 5) + b"\0" * 18
    phrases = _decode_phrase_file(_file(_tag(b"PSSI", content_header + entry_a + entry_b)), BeatGrid(0, 128))
    assert [(p.beat_start, p.beat_end, p.label) for p in phrases] == [(1, 33, "Intro"), (33, 65, "Chorus")]


def test_pssi_decoder_recovers_after_invalid_tag():
    invalid = struct.pack(">4sII", b"PWV5", 24, 0xFFFFFFF0) + b"bad" + b"\0" * 20
    content_header = struct.pack(">IHH", 24, 1, 1) + b"\0" * 6 + struct.pack(">H", 33) + b"\0" * 2 + b"\0\0"
    entry = struct.pack(">HHH", 0, 1, 1) + b"\0" * 18
    phrases = _decode_phrase_file(_file(invalid + _tag(b"PSSI", content_header + entry)), BeatGrid(0, 128))
    assert len(phrases) == 1
    assert phrases[0].label == "Intro"
