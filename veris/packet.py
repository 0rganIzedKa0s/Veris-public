"""The contract between tiers.

Every fast model writes into one Packet; every slow model reads from it. The
packet is the only thing that crosses a tier boundary, so each tier can be
trained, benchmarked and replaced on its own.

    gate    -> Gate        (is this frame worth looking at; indoor/outdoor/text votes)
    reflex  -> Obj[]       (what is where, how far, and is it closing)
    reader  -> TextLine[]  (exact words with confidence)
    planner -> Situation   (where the user is and what matters here)

`Packet.to_prompt(mode)` renders the slice a given expert was trained on. The
format is deliberately terse: it is prefilled on every frame, and prefill is
what time-to-first-token is made of.

Geometry lives here, not in the VLM. Bearing comes from the pixel column,
range from the ground plane, closing speed from box growth. The expert
narrates a scene whose geometry is already measured, which is the part a 3B
model gets wrong on its own (base laterality: 86%).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("outdoor", "indoor", "reader", "nav")

# Below this OCR confidence a word is announced as unreadable, never guessed.
LEGIBLE_CONF = 0.99   # evals/results/reader_ocr.json: at 0.99, 97% of words are within one character
# OCR's words are passed to the reader as hints only when it could read at
# least this share of the lines. Below it the hint is mostly "[unreadable]",
# and that talks the reader out of text it can read from the pixels. Above it
# the hints cut invented words (evals/results/reader_fovea.json, hint sweep).
HINT_MIN_LEGIBLE = 0.3

# The reflex tier warns from 0.45 confidence: a missed pole costs more than a
# false warning. A description is the reverse: the expert can see the image,
# and repeats a false object it is handed. Its packet lists only detections
# from here up (experts_all_boxes.json: 19 -> 14 invented objects per 100
# outdoor images, coverage unchanged).
PACKET_MIN_CONF = 0.7


def clock_from_bearing(bearing_deg: float) -> int:
    """Signed bearing (negative = left, 0 = straight ahead) -> clock hour.
    30 degrees per hour, clamped to the forward half: 9, 10, 11, 12, 1, 2, 3."""
    h = round(max(-90.0, min(90.0, bearing_deg)) / 30.0)
    return 12 if h == 0 else (h if h > 0 else 12 + h)


# Things that stand on streets. A scene classifier can be fooled by a wall or
# an awning; a detected car or traffic light is harder to argue with. Seeing
# one pulls the gate's indoor vote down by STREET_PULL: enough to keep a
# lukewarm "indoor" (0.7-0.8) from switching the mode, not enough to turn a
# confident one (0.95, a car seen through a window) into "outdoor".
STREET_CLASSES = frozenset({"car", "bus", "truck", "traffic light", "traffic sign",
                            "stop sign", "street light", "fire hydrant"})
STREET_PULL = 0.3


def indoor_with_objects(indoor_p: float, labels) -> float:
    """The gate's indoor vote, corrected by what the reflex detector sees."""
    return max(0.0, indoor_p - STREET_PULL) if STREET_CLASSES & set(labels) else indoor_p


def side_from_bearing(bearing_deg: float, dead_band_deg: float = 8.0) -> str:
    if abs(bearing_deg) <= dead_band_deg:
        return "ahead"
    return "left" if bearing_deg < 0 else "right"


@dataclass(frozen=True)
class Gate:
    """Tier 0. Runs on every frame before anything expensive."""

    usable: bool = True
    issue: str | None = None        # blur | dark | bright | obstructed
    indoor_p: float = 0.5
    text_p: float = 0.0
    sharpness: float = 1.0          # 0..1, used to pick the best of N frames


@dataclass(frozen=True)
class Obj:
    """Tier 1. One tracked object, in user-relative coordinates."""

    label: str
    bearing_deg: float
    conf: float = 1.0
    dist_m: float | None = None     # ground-plane range; None when not measurable
    ttc_s: float | None = None      # time to contact from looming; None if not closing
    in_path: bool = False           # inside the walking corridor
    track_id: int | None = None

    @property
    def clock(self) -> int:
        return clock_from_bearing(self.bearing_deg)

    def phrase(self, clock_face: bool = True) -> str:
        where = (f"{self.clock} o'clock" if clock_face
                 else side_from_bearing(self.bearing_deg))
        parts = [self.label, where]
        if self.dist_m is not None:
            parts.append(f"{self.dist_m:.0f} m" if self.dist_m >= 2
                         else "within reach")
        if self.ttc_s is not None:
            parts.append("approaching")
        if self.in_path:
            parts.append("(in path)")
        return " ".join(parts)


@dataclass(frozen=True)
class TextLine:
    """Tier 1b. One OCR line in reading order."""

    text: str
    conf: float
    bearing_deg: float = 0.0
    height_frac: float = 0.0        # glyph height / frame height: prominence
    group: int = 0                  # lines sharing a group belong to one sign/page

    @property
    def legible(self) -> bool:
        return self.conf >= LEGIBLE_CONF and bool(self.text.strip())


@dataclass(frozen=True)
class Situation:
    """The planner's output: the slow LLM's instructions to the fast models.

    Field names match the existing context engine so cloud and offline
    planners are interchangeable. `class_priors` is the part that reaches
    below the VLM: it re-weights which detector classes are worth speaking
    about at this location (a crossing raises traffic lights, a station
    raises stairs)."""

    situation: str = "unknown"
    suggested_mode: str = "outdoor_nav"   # arbiter.MODES vocabulary
    look_for: tuple[str, ...] = ()
    context_text: str = ""
    source: str = "none"            # cloud | offline_llm | offline_rules | none
    class_priors: dict[str, float] = field(default_factory=dict)
    route_hint: str | None = None   # next instruction parsed from the phone


@dataclass(frozen=True)
class Packet:
    t: float = 0.0
    gate: Gate = field(default_factory=Gate)
    objects: tuple[Obj, ...] = ()
    text: tuple[TextLine, ...] = ()
    situation: Situation = field(default_factory=Situation)
    moving: bool = False

    def ranked_objects(self, limit: int = 6) -> list[Obj]:
        """Most urgent first: closing, then in path, then nearest, weighted by
        the planner's class priors."""
        pri = self.situation.class_priors

        def key(o: Obj):
            return (
                o.ttc_s if o.ttc_s is not None else 1e9,
                not o.in_path,
                (o.dist_m if o.dist_m is not None else 50.0) / pri.get(o.label, 1.0),
            )

        return sorted(self.objects, key=key)[:limit]

    def to_prompt(self, mode: str, clock_face: bool = True) -> str:
        """The text block an expert sees after the image. Empty sections are
        omitted so the model never learns to read meaning into a blank."""
        lines: list[str] = []
        sure = [o for o in self.ranked_objects(len(self.objects)) if o.conf >= PACKET_MIN_CONF][:6]
        if mode in ("outdoor", "indoor", "nav") and sure:
            objs = "; ".join(o.phrase(clock_face) for o in sure)
            lines.append(f"objects: {objs}")
        if mode in ("reader", "outdoor", "indoor") and self.text:
            limit = None if mode == "reader" else 3
            shown = self.text[:limit]
            words = " | ".join(f'"{t.text}"' if t.legible else "[unreadable]"
                               for t in shown)
            lines.append(f"text: {words}")
        s = self.situation
        if s.source != "none" and mode != "reader":
            place = s.situation.replace("_", " ")
            if s.look_for:
                place += "; if visible: " + ", ".join(s.look_for)
            if s.route_hint:
                place += f"; route: {s.route_hint}"
            lines.append(f"place: {place} [GPS]")
        if not lines:
            return ""
        return "<sense>\n" + "\n".join(lines) + "\n</sense>"


def demo() -> None:
    assert indoor_with_objects(0.75, ["person", "car"]) == 0.45
    assert indoor_with_objects(0.75, ["person", "chair"]) == 0.75
    assert [clock_from_bearing(b) for b in (-90, -60, -30, 0, 30, 60, 90)] == \
        [9, 10, 11, 12, 1, 2, 3]
    assert clock_from_bearing(-200) == 9 and clock_from_bearing(14) == 12
    assert side_from_bearing(-20) == "left" and side_from_bearing(3) == "ahead"

    p = Packet(
        objects=(
            Obj("car", -55, dist_m=8),
            Obj("pole", 25, dist_m=1.4, in_path=True),
            Obj("person", 2, dist_m=3, ttc_s=2.1),
        ),
        text=(TextLine("PUSH", 0.995), TextLine("xq7", 0.2)),
        situation=Situation("street_crossing", "outdoor_nav", ("a walk signal",),
                            source="offline_rules",
                            class_priors={"traffic light": 3.0}),
    )
    # closing object first, then the in-path pole, then the far car
    assert [o.label for o in p.ranked_objects()] == ["person", "pole", "car"]
    out = p.to_prompt("outdoor")
    assert "person 12 o'clock 3 m approaching" in out
    assert "pole 1 o'clock within reach (in path)" in out
    assert '"PUSH" | [unreadable]' in out and "xq7" not in out
    assert "street crossing; if visible: a walk signal [GPS]" in out
    # the reader never sees place context: GPS must not leak into read text
    assert "GPS" not in p.to_prompt("reader")
    assert Packet().to_prompt("outdoor") == ""
    # an unsure detection warns (reflex) but is not handed to a describer
    assert "truck" not in Packet(objects=(Obj("truck", 5, conf=0.5), Obj("car", -20, conf=0.9))).to_prompt("outdoor")
    print(out)


if __name__ == "__main__":
    demo()
