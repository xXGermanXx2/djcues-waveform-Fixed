import struct

from djcues.db import _decode_waveform_file, _decode_waveform_tag


def _tag(tag_type: bytes, payload: bytes, header_len: int = 24) -> bytes:
    header = struct.pack(">4sII", tag_type, header_len, header_len + len(payload))
    return header + (b"\0" * (header_len - 12)) + payload


def _file(*tags: bytes) -> bytes:
    header_len = 32
    body = b"".join(tags)
    return struct.pack(">4sIII", b"ANLZ", header_len, header_len + len(body), 0) + (b"\0" * 16) + body


def test_decodes_pwv5_packed_color_detail():
    # content header: entry bytes=2, count=2, unknown=0x00960305
    values = [(1 << 12) | (2 << 9) | (3 << 6) | (31 << 2), (7 << 12) | (6 << 9) | (5 << 6) | (8 << 2)]
    payload = struct.pack(">IIIHH", 2, 2, 0x00960305, *values)
    points = _decode_waveform_file(_file(_tag(b"PWV5", payload)), max_points=800)
    assert len(points) == 2
    assert (points[0].red, points[0].green, points[0].blue) == (1, 2, 3)
    assert points[0].height == 1.0


def test_decodes_pwv7_three_band_detail():
    payload = struct.pack(">III6B", 3, 2, 0x00960000, 10, 40, 80, 20, 50, 90)
    points = _decode_waveform_file(_file(_tag(b"PWV7", payload)), max_points=800)
    assert len(points) == 2
    assert points[0].height == 80 / 255
    assert points[0].red > points[0].blue


def test_skips_invalid_tag_and_finds_following_waveform():
    invalid = struct.pack(">4sII", b"PSSI", 32, 0xFFFFFFF0) + b"bad" + b"\0" * 20
    payload = struct.pack(">IIIH", 2, 1, 0x00960305, (4 << 12) | (2 << 9) | (1 << 6) | (16 << 2))
    points = _decode_waveform_file(_file(invalid + _tag(b"PWV5", payload)), max_points=800)
    assert points


def test_pwv5_stale_count_uses_available_payload():
    payload = struct.pack(">IIIH", 2, 999999, 0x00960305, 0x7C00)
    points = _decode_waveform_tag("PWV5", 24, _tag(b"PWV5", payload))
    assert len(points) == 1
