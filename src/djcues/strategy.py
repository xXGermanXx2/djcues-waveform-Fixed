"""Cue placement strategy engine — maps PSSI phrases to cue points."""

from __future__ import annotations

from djcues.constants import CUE_SYSTEM, CUE_SYSTEM_BY_PAD
from djcues.models import BeatGrid, CuePoint, CueProposal, Phrase, Track


def _spectral_similarity(wf_points: list, i0: int, i_mid: int, i1: int) -> float:
    """Compare the RGB (bass/mid/treble) profile of two halves of a waveform section.

    Returns a similarity score from 0.0 (completely different) to 1.0 (identical).
    Uses mean-squared-error of the per-point RGB values, normalized.
    """
    half_len = min(i_mid - i0, i1 - i_mid)
    if half_len < 2:
        return 0.0

    first_half = wf_points[i0 : i0 + half_len]
    second_half = wf_points[i_mid : i_mid + half_len]

    total_mse = 0.0
    for a, b in zip(first_half, second_half):
        # RGB values are 0-7, normalize to 0-1
        dr = (a.red - b.red) / 7
        dg = (a.green - b.green) / 7
        db = (a.blue - b.blue) / 7
        dh = a.height - b.height  # already 0-1
        total_mse += dr * dr + dg * dg + db * db + dh * dh

    # Max possible MSE per point = 4 (all channels off by 1.0)
    mse = total_mse / half_len / 4
    return max(0.0, 1.0 - mse)


def _find_stable_loop(
    track: Track,
    search_start_ms: float,
    search_end_ms: float,
    bar_sizes: tuple[int, ...] = (8, 4, 2, 1),
    min_energy: float = 0.05,
    min_similarity: float = 0.7,
) -> tuple[float, int, float] | None:
    """Find a stable, loopable region using waveform energy and spectral similarity.

    Scans bar-aligned windows from search_start_ms forward. Tries 8 bars first,
    then falls back to 4, 2, 1. A good loop has non-trivial energy AND the
    second half spectrally matches the first half (so the loop repeats cleanly).

    Returns (position_ms, loop_bars, similarity_score) or None if nothing found.
    """
    if not track.waveform:
        return None

    bg = track.beat_grid
    n = len(track.waveform)
    total_ms = track.duration_ms
    if total_ms <= 0 or n == 0:
        return None

    for loop_bars in bar_sizes:
        loop_ms = bg.bars_to_ms(loop_bars)
        best_pos = None
        best_score = 0.0

        # Snap search start to bar boundary
        start_beat = bg.ms_to_beat(search_start_ms)
        bar_start = ((start_beat - 1) // 4) * 4 + 1
        pos_ms = bg.beat_to_ms(bar_start)

        while pos_ms + loop_ms <= search_end_ms and pos_ms + loop_ms <= total_ms:
            # Map to waveform indices
            i0 = int(n * pos_ms / total_ms)
            i1 = int(n * (pos_ms + loop_ms) / total_ms)
            i_mid = (i0 + i1) // 2
            if i1 - i0 < 4:
                pos_ms += bg.bars_to_ms(1)
                continue

            # Check energy
            heights = [p.height for p in track.waveform[i0:i1]]
            mean_e = sum(heights) / len(heights)
            if mean_e < min_energy:
                pos_ms += bg.bars_to_ms(1)
                continue

            # Check spectral similarity between halves
            similarity = _spectral_similarity(track.waveform, i0, i_mid, i1)
            if similarity > best_score:
                best_score = similarity
                best_pos = pos_ms

            pos_ms += bg.bars_to_ms(1)

        if best_pos is not None and best_score >= min_similarity:
            return (best_pos, loop_bars, best_score)

            pos_ms += bg.bars_to_ms(1)  # slide by 1 bar

    return None


class CueStrategy:
    """Proposes cue placements based on phrase analysis and the cue system."""

    def __init__(
        self,
        memory_offset_bars: int = 16,
        loop_length_bars: int = 4,
    ) -> None:
        self.memory_offset_bars = memory_offset_bars
        self.loop_length_bars = loop_length_bars

    def propose(self, track: Track) -> CueProposal:
        """Generate a cue proposal for a track based on its phrase structure."""
        bg = track.beat_grid
        phrases = track.phrases
        hot_cues: list[CuePoint] = []
        memory_cues: list[CuePoint] = []
        confidence: dict[str, float] = {}
        notes: list[str] = []

        # Build positions dict keyed by pad letter
        positions: dict[str, float] = {}

        # --- A: First Beat ---
        positions["A"] = bg.beat_to_ms(1)
        confidence["A"] = 1.0
        notes.append("A (First Beat): beat 1")

        # --- B: Bassline Introduction ---
        # Rekordbox does not expose a dedicated bassline-onset label.
        # Use the first energetic Up phrase after the intro; for other
        # phrase vocabularies, use the first non-intro segment.
        intro_end = max((p.position_ms + p.duration_ms for p in phrases if p.label == "Intro"), default=positions["A"])
        bass_candidates = [p for p in phrases if p.position_ms >= intro_end and p.label == "Up"]
        if not bass_candidates:
            bass_candidates = [p for p in phrases if p.position_ms > positions["A"] and p.label not in ("Intro", "Unknown")]
        if bass_candidates:
            bass_phrase = bass_candidates[0]
            positions["B"] = bass_phrase.position_ms
            confidence["B"] = 0.65
            notes.append(f"B (Bassline Introduction): {bass_phrase.label} at beat {bass_phrase.beat_start}")
        else:
            positions["B"] = positions["A"]
            confidence["B"] = 0.25
            notes.append("B (Bassline Introduction): no post-intro phrase found — using beat 1")

        # --- Main drop detection ---
        # Rekordbox's Chorus phrase is the best available proxy for a drop.
        # Ignore very early choruses so an intro/mini-hook is not mistaken
        # for the main drop.
        choruses = [p for p in phrases if p.label == "Chorus"]
        min_drop_ms = track.duration_ms * 0.20
        late_choruses = [p for p in choruses if p.position_ms >= min_drop_ms]
        if late_choruses:
            main_drop = late_choruses[0]
            positions["G"] = main_drop.position_ms
            confidence["G"] = 0.85
            notes.append(f"G (Main Drop): first Chorus after 20% at beat {main_drop.beat_start}")
        elif choruses:
            main_drop = choruses[-1]
            positions["G"] = main_drop.position_ms
            confidence["G"] = 0.60
            notes.append(f"G (Main Drop): fallback to last Chorus at beat {main_drop.beat_start}")
        else:
            main_drop = None
            confidence["G"] = 0.0
            notes.append("G (Main Drop): no Chorus found — skipped")

        # --- A/B/C/D/E/F markers around the main drop ---
        # C is the first breakdown before the main drop.
        before_drop = [p for p in phrases if main_drop is not None and p.position_ms < main_drop.position_ms]
        early_breakdowns = [p for p in before_drop if p.label in ("Down", "Bridge")]
        if early_breakdowns:
            first_breakdown = early_breakdowns[0]
            positions["C"] = first_breakdown.position_ms
            confidence["C"] = 0.80
            notes.append(f"C (First Breakdown): {first_breakdown.label} at beat {first_breakdown.beat_start}")
        else:
            pre_drop_up = [p for p in before_drop if p.label in ("Up", "Verse1", "Verse2", "Verse3", "Verse4", "Verse5", "Verse6")]
            if pre_drop_up:
                fallback = pre_drop_up[-1]
                positions["C"] = fallback.position_ms
                confidence["C"] = 0.45
                notes.append(f"C (First Breakdown): no Down/Bridge; fallback to {fallback.label} at beat {fallback.beat_start}")
            else:
                confidence["C"] = 0.0
                notes.append("C (First Breakdown): no suitable pre-drop phrase found")

        if main_drop is not None:
            # D/F are exact beat-grid waypoints before the main drop.
            positions["D"] = max(bg.beat_to_ms(1), main_drop.position_ms - bg.ms_per_beat * 8)
            positions["F"] = max(bg.beat_to_ms(1), main_drop.position_ms - bg.ms_per_beat * 32)
            confidence["D"] = 0.95
            confidence["F"] = 0.95
            notes.append("D (8 Beats Before Main Drop): exactly 8 beats before G")
            notes.append("F (32 Beats Before Main Drop): exactly 32 beats before G")

            # E is the first substantial breakdown after the main drop.
            after_drop = [p for p in phrases if p.position_ms > main_drop.position_ms and p.label in ("Down", "Bridge")]
            if after_drop:
                main_breakdown = max(after_drop, key=lambda p: (p.duration_ms, -p.position_ms))
                positions["E"] = main_breakdown.position_ms
                confidence["E"] = 0.85
                notes.append(f"E (Main Breakdown): {main_breakdown.label} at beat {main_breakdown.beat_start}")
            else:
                confidence["E"] = 0.0
                notes.append("E (Main Breakdown): no Down/Bridge found after Main Drop")
        else:
            confidence["D"] = confidence["F"] = 0.0
            notes.append("D/F: no Main Drop available for beat-offset markers")

        # H is the final major Chorus, or the Outro when no later Chorus exists.
        if choruses:
            final_drop = choruses[-1]
            positions["H"] = final_drop.position_ms
            confidence["H"] = 0.85
            notes.append(f"H (Final Drop / Outro): final Chorus at beat {final_drop.beat_start}")
        elif phrases:
            positions["H"] = next((p.position_ms for p in phrases if p.label == "Outro"), phrases[-1].position_ms)
            confidence["H"] = 0.55
            notes.append("H (Final Drop / Outro): no Chorus found — using Outro/last phrase")
        else:
            confidence["H"] = 0.0
            notes.append("H (Final Drop / Outro): no phrases found")

        # --- Build CuePoint objects ---
        for slot in CUE_SYSTEM:
            pad = slot.pad
            if pad not in positions:
                continue

            pos_ms = positions[pad]
            loop_end = None
            if slot.is_loop:
                loop_end = pos_ms + bg.bars_to_ms(self.loop_length_bars)

            hot_cues.append(CuePoint(
                kind=slot.kind,
                position_ms=pos_ms,
                loop_end_ms=loop_end,
                color_table_index=slot.hot_cue_color_table_index,
                color=slot.hot_cue_color,
                comment=slot.hot_cue_label,
            ))

            # Memory cue
            if slot.memory_offset_bars == 0:
                mem_pos = pos_ms
            else:
                mem_pos = pos_ms - bg.bars_to_ms(self.memory_offset_bars)
                first_beat_ms = bg.beat_to_ms(1)
                if mem_pos < first_beat_ms:
                    mem_pos = first_beat_ms
                else:
                    # Snap to nearest downbeat (bar start)
                    mem_beat = bg.ms_to_beat(mem_pos)
                    bar_beat = ((mem_beat - 1) // 4) * 4 + 1
                    mem_pos = bg.beat_to_ms(bar_beat)

            mem_loop_end = None
            if slot.is_loop:
                mem_loop_end = mem_pos + bg.bars_to_ms(self.loop_length_bars)

            memory_cues.append(CuePoint(
                kind=0,
                position_ms=mem_pos,
                loop_end_ms=mem_loop_end,
                color_table_index=slot.memory_cue_color_table_index,
                color=slot.memory_cue_color,
                comment=slot.memory_cue_label,
            ))

        return CueProposal(
            track=track,
            hot_cues=hot_cues,
            memory_cues=memory_cues,
            confidence=confidence,
            notes=notes,
        )
