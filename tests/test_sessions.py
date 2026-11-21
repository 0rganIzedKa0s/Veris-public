"""Unit tests for the session-data flywheel: rating->generation join, dedup,
and the sync transfer parser.

  uv run python -m pytest tests/test_sessions.py -q
"""

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.build_sft_set import (  # noqa: E402
    collect_session_data,
    frame_hash,
    join_ratings,
    load_session_records,
)
from dataset.sync_sessions import parse_transfers  # noqa: E402


# --- helpers ---------------------------------------------------------------

def gen(frame, output, mode="describe", question="", context="", ts=0.0):
    return {"ts": ts, "frame": frame, "mode": mode, "context": context,
            "question": question, "variants": "", "output": output,
            "ttft_s": 1.0, "total_s": 2.0}


def rating(output, r="up", note=None, ts=0.0):
    rec = {"kind": "rating", "output": output, "rating": r, "ts": ts}
    if note is not None:
        rec["note"] = note
    return rec


def write_session(dirpath: Path, name: str, records) -> Path:
    p = dirpath / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


def write_frame(dirpath: Path, name: str, color=(200, 30, 30)) -> Path:
    p = dirpath / name
    Image.new("RGB", (8, 8), color).save(p, format="JPEG")
    return p


# --- load_session_records ---------------------------------------------------

def test_load_skips_corrupt_and_blank_lines(tmp_path):
    p = tmp_path / "session-1.jsonl"
    p.write_text(json.dumps(gen("f.jpg", "A door ahead.")) + "\n"
                 + "\n{not json...\n"
                 + json.dumps(rating("A door ahead.")) + "\n"
                 + '{"truncated": ')  # live-appended file cut mid-write
    recs = load_session_records(p)
    assert len(recs) == 2
    assert recs[0]["output"] == "A door ahead."
    assert recs[1]["kind"] == "rating"


# --- join_ratings -----------------------------------------------------------

def test_join_matches_nearest_preceding_generation_by_output():
    records = [
        gen("a.jpg", "A blue bus on your left."),
        gen("b.jpg", "A red door ahead."),
        rating("A red door ahead.", "up"),
    ]
    joined, unmatched = join_ratings(records)
    assert unmatched == 0
    assert len(joined) == 1
    g, fbs = joined[0]
    assert g["frame"] == "b.jpg"
    assert fbs[0]["rating"] == "up"


def test_join_reaches_past_newer_generation_when_output_matches_older():
    # user rates the previous description after a new frame already streamed
    records = [
        gen("a.jpg", "A blue bus on your left."),
        gen("b.jpg", "A red door ahead."),
        rating("A blue bus on your left.", "down"),
    ]
    joined, _ = join_ratings(records)
    assert len(joined) == 1
    assert joined[0][0]["frame"] == "a.jpg"


def test_join_repeated_identical_outputs_picks_nearest():
    records = [
        gen("a.jpg", "A blue bus on your left."),
        gen("b.jpg", "A blue bus on your left."),
        rating("A blue bus on your left.", "up"),
    ]
    joined, _ = join_ratings(records)
    assert len(joined) == 1
    assert joined[0][0]["frame"] == "b.jpg"


def test_join_null_output_rating_joins_nearest_preceding():
    records = [
        gen("a.jpg", "A door."),
        {"kind": "rating", "output": None, "rating": "up", "ts": 1.0},
    ]
    joined, unmatched = join_ratings(records)
    assert unmatched == 0
    assert joined[0][0]["frame"] == "a.jpg"


def test_join_rating_before_any_generation_is_unmatched():
    records = [rating("Something never generated.", "up"), gen("a.jpg", "A door.")]
    joined, unmatched = join_ratings(records)
    assert joined == []
    assert unmatched == 1


def test_join_groups_multiple_feedback_on_one_generation():
    records = [
        gen("a.jpg", "A door."),
        rating("A door.", "down"),
        {"kind": "rating", "output": "A door.", "rating": None,
         "note": "say which side", "ts": 2.0},
    ]
    joined, _ = join_ratings(records)
    assert len(joined) == 1
    assert len(joined[0][1]) == 2


# --- collect_session_data: routing ------------------------------------------

def test_thumbs_up_becomes_verbatim_sft_and_down_goes_to_queue(tmp_path):
    write_frame(tmp_path, "up.jpg", (10, 200, 10))
    write_frame(tmp_path, "down.jpg", (10, 10, 200))
    write_session(tmp_path, "session-1.jsonl", [
        gen("up.jpg", "A door on your left.", question="Where is the door?"),
        rating("A door on your left.", "up"),
        gen("down.jpg", "The room is a home office."),
        rating("The room is a home office.", "down",
               note="it's a bedroom, not an office"),
    ])
    ups, queue, counts = collect_session_data(tmp_path, set())
    assert counts["sft"] == 1 and counts["queued"] == 1
    assert ups[0]["target"] == "A door on your left."  # verbatim
    assert ups[0]["user_text"].startswith("Where is the door?")
    q = queue[0]
    assert q["frame"] == "down.jpg"
    assert q["rating"] == "down"
    assert q["notes"] == ["it's a bedroom, not an office"]
    assert q["frame_sha256"] == frame_hash(tmp_path / "down.jpg")


def test_note_only_send_goes_to_queue(tmp_path):
    write_frame(tmp_path, "a.jpg")
    write_session(tmp_path, "session-1.jsonl", [
        gen("a.jpg", "A chair ahead."),
        {"kind": "rating", "output": "A chair ahead.", "rating": None,
         "note": "mention the table too", "ts": 1.0},
    ])
    ups, queue, counts = collect_session_data(tmp_path, set())
    assert ups == []
    assert counts["queued"] == 1
    assert queue[0]["rating"] is None
    assert queue[0]["notes"] == ["mention the table too"]


def test_unrated_generations_are_ignored(tmp_path):
    write_frame(tmp_path, "a.jpg")
    write_session(tmp_path, "session-1.jsonl", [
        gen("a.jpg", "A chair ahead."),
        gen("a.jpg", "A table ahead."),
    ])
    ups, queue, counts = collect_session_data(tmp_path, set())
    assert ups == [] and queue == []
    assert counts["generations"] == 2 and counts["feedback"] == 0


def test_missing_frame_is_skipped(tmp_path):
    write_session(tmp_path, "session-1.jsonl", [
        gen("gone.jpg", "A chair ahead."),
        rating("A chair ahead.", "up"),
    ])
    ups, queue, counts = collect_session_data(tmp_path, set())
    assert ups == [] and queue == []
    assert counts["missing_frame"] == 1


# --- collect_session_data: dedup ---------------------------------------------

def test_dedup_same_frame_bytes_and_question_across_sessions(tmp_path):
    # identical frame content under two names (the Orin cam repeats frames)
    write_frame(tmp_path, "f1.jpg", (77, 77, 77))
    (tmp_path / "f2.jpg").write_bytes((tmp_path / "f1.jpg").read_bytes())
    write_session(tmp_path, "session-1.jsonl", [
        gen("f1.jpg", "A blue bus on your left."),
        rating("A blue bus on your left.", "up"),
    ])
    write_session(tmp_path, "session-2.jsonl", [
        gen("f2.jpg", "A blue bus on your left."),
        rating("A blue bus on your left.", "up"),
    ])
    ups, _, counts = collect_session_data(tmp_path, set())
    assert counts["sft"] == 1
    assert counts["dup"] == 1
    assert len(ups) == 1


def test_same_frame_different_question_is_not_a_dup(tmp_path):
    write_frame(tmp_path, "f1.jpg", (77, 77, 77))
    (tmp_path / "f2.jpg").write_bytes((tmp_path / "f1.jpg").read_bytes())
    write_session(tmp_path, "session-1.jsonl", [
        gen("f1.jpg", "Yes.", question="Is there a bus?"),
        rating("Yes.", "up"),
        gen("f2.jpg", "It is blue.", question="What color is the bus?"),
        rating("It is blue.", "up"),
    ])
    _, _, counts = collect_session_data(tmp_path, set())
    assert counts["sft"] == 2 and counts["dup"] == 0


def test_dedup_against_existing_training_data_via_seeded_keys(tmp_path):
    write_frame(tmp_path, "f1.jpg", (5, 5, 5))
    write_session(tmp_path, "session-1.jsonl", [
        gen("f1.jpg", "A pole ahead.", question="What is ahead?"),
        rating("A pole ahead.", "up"),
    ])
    # seed `seen` as if this exact (frame, question) is already in the SFT set
    seen = {(frame_hash(tmp_path / "f1.jpg"), "What is ahead?")}
    ups, queue, counts = collect_session_data(tmp_path, seen)
    assert ups == [] and queue == []
    assert counts["dup"] == 1


def test_dedup_applies_to_rewrite_queue_too(tmp_path):
    write_frame(tmp_path, "f1.jpg", (9, 9, 9))
    (tmp_path / "f2.jpg").write_bytes((tmp_path / "f1.jpg").read_bytes())
    write_session(tmp_path, "session-1.jsonl", [
        gen("f1.jpg", "A wall.", ),
        rating("A wall.", "down", note="wrong"),
        gen("f2.jpg", "A wall."),
        rating("A wall.", "down", note="still wrong"),
    ])
    _, queue, counts = collect_session_data(tmp_path, set())
    assert len(queue) == 1 and counts["dup"] == 1


# --- sync_sessions transfer parsing ------------------------------------------

def test_parse_transfers_splits_logs_and_frames():
    out = "./\nsession-123.jsonl\nframe-1.jpg\nframe-2.jpg\nnotes.txt\n\n"
    logs, frames = parse_transfers(out)
    assert logs == ["session-123.jsonl"]
    assert frames == ["frame-1.jpg", "frame-2.jpg"]
