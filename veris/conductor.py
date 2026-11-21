"""The conductor: which tier speaks, and when.

One frame flows down the tiers and stops at the first one that can answer:

    gate     bad frame            -> dropped; no later tier ever sees it
    reflex   hazard               -> spoken now, from a template, in any mode
    reader   short legible text   -> spoken from OCR, no language model
    expert   everything else      -> one adapter on the shared base, primed
                                     with the packet the fast tiers filled

Switching is the arbiter's job (veris/arbiter.py): user command, then vision
votes, then the planner's proposal, behind dwell and residence hysteresis.
The conductor adds three things on top of it:

- it feeds the arbiter the gate's votes and the planner's situation;
- it keeps the language model's cache on the expert in use: warmed at boot
  and on the first frame after a switch, never for a candidate still in dwell;
- it knows when a description is worth giving unasked: the set of things in
  the user's way changed, not a timer fired.

Every collaborator is injected, so the same class runs on the board or under
the stubs in demo().
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from .arbiter import Arbiter, Motion
from .experts import EXPERTS, expert_for
from .packet import HINT_MIN_LEGIBLE, Gate, Packet, Situation, indoor_with_objects

WARM_GAP_S = 2.0        # a wavering gate vote must not become a warm-up on every frame
FRAME_BUFFER = 5          # describe() picks the sharpest of the last N usable frames
BEST_OF_WINDOW_S = 0.6    # ...and no older than this: the head may have turned since
MIN_DESCRIBE_GAP_S = 8.0  # unprompted descriptions are never closer than this
NEAR_M = 6.0              # an object nearer than this is part of "the scene"

HOLD = {"blur": "Too blurry. Hold still.", "dark": "Too dark to see.",
        "bright": "Too much glare.", "obstructed": "Something is covering the lens.",
        None: "I can't see clearly yet."}


@dataclass(frozen=True)
class Say:
    kind: str       # hazard | text | describe | cue
    text: str
    source: str     # reflex | ocr | expert:<name> | gate


def _jpeg(bgr, side: int) -> bytes:
    import cv2
    s = side / max(bgr.shape[:2]) if side else 1     # side 0: send as is
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


class Conductor:
    def __init__(self, arbiter: Arbiter, gate, reflex, reader, planner, llm,
                 now: Callable[[], float], speak_sure: Callable | None = None,
                 warm: Callable[[str], None] | None = None,
                 encode: Callable = _jpeg, clock_face: bool = False,
                 foveate: Callable | None = None, doc_sheet: Callable | None = None):
        self.arb, self.gate, self.reflex = arbiter, gate, reflex
        self.reader, self.planner, self.llm, self.now = reader, planner, llm, now
        self.speak_sure, self.warm, self.encode = speak_sure, warm, encode
        self.clock_face, self.foveate, self.doc_sheet = clock_face, foveate, doc_sheet
        self.situation = Situation()
        self.packet = Packet()
        self.last_gate = Gate()
        self._frames: deque = deque(maxlen=FRAME_BUFFER)   # (bgr, sharpness, time)
        self._warmed: str | None = None
        self._warm_t = -1e9
        # a warm-up never holds a camera frame: the reflex loop keeps its rate
        self._spawn = lambda fn, *a: threading.Thread(target=fn, args=a, daemon=True).start()
        self._described_t = -1e9
        self._described_sig: tuple | None = None

    # ── slow input: phone GPS, every 30-60 s ─────────────────────────
    def on_fix(self, fix: dict, route=None) -> Situation:
        self.situation = self.planner.plan(fix, route)
        self.arb.update_context(self.situation.suggested_mode, self.situation.situation)
        return self.situation

    # ── fast input: every camera frame ───────────────────────────────
    def on_frame(self, bgr) -> list[Say]:
        g = self.last_gate = self.gate.run(bgr)
        if not g.usable:
            return []                       # nothing downstream sees a bad frame
        t = self.now()
        self._frames.append((bgr, g.sharpness, self.now()))
        objs, say = self.reflex.step(bgr, t, self.situation)
        # two tiers vote on indoor/outdoor: the gate's scene guess, corrected
        # by street objects the detector actually sees
        g = self.last_gate = replace(g, indoor_p=indoor_with_objects(
            g.indoor_p, [o.label for o in objs]))
        self.arb.update_gate(g.indoor_p, g.text_p)
        self.packet = Packet(t, g, tuple(objs), (), self.situation,
                             moving=self.arb.motion is Motion.WALKING)
        self._prewarm(bgr)
        return [Say("hazard", say, "reflex")] if say else []

    def current_expert(self) -> str:
        return expert_for(self.arb.mode, self.last_gate.indoor_p)

    def _prewarm(self, bgr=None) -> None:
        """Keep the language model's cache on the expert in use: warmed at
        boot and on the first frame after any switch. A candidate still
        waiting out its dwell is not warmed: that takes the cache away from
        the expert the user can still ask (measured: 1.5 s against 1.2 s)."""
        name = self.current_expert()
        if not self.warm or name == self._warmed or self.now() - self._warm_t < WARM_GAP_S:
            return
        boot = self._warmed is None         # the first image through the vision tower is slow too
        self._warmed, self._warm_t = name, self.now()
        if boot and bgr is not None:
            self._spawn(self.warm, name, self.encode(bgr, EXPERTS[name].image_side))
        else:
            self._spawn(self.warm, name)

    # ── when to describe unasked ─────────────────────────────────────
    def _signature(self) -> tuple:
        near = {o.label for o in self.packet.objects
                if o.in_path or (o.dist_m is not None and o.dist_m < NEAR_M)}
        return (self.arb.mode, self.situation.situation, tuple(sorted(near)))

    def scene_changed(self) -> bool:
        """True when the scene differs from the one last described and enough
        time has passed. A steady scene stays silent however long it lasts."""
        if self.now() - self._described_t < MIN_DESCRIBE_GAP_S:
            return False
        return self._signature() != self._described_sig

    # ── on demand, or when scene_changed() ───────────────────────────
    def describe(self, question: str | None = None) -> Iterator[Say]:
        # sharpest recent frame; an older one may show where the head was pointing before
        recent = [f for f in self._frames if self.now() - f[2] <= BEST_OF_WINDOW_S]
        if not recent:
            yield Say("cue", HOLD[self.last_gate.issue], "gate")
            return
        frame = max(recent, key=lambda f: f[1])[0]
        name = self.current_expert()
        pk = self.packet
        side = EXPERTS[name].image_side
        pre = self.reader.layout(frame) if name == "reader" and self.doc_sheet else None
        doc = self.doc_sheet(frame, pre) if pre else None
        if doc is not None:
            # a page, a menu, a receipt: the lines are found but not recognised;
            # the base model reads the cropped page (reader.document_sheet)
            name, frame, side, pk = "document", doc, 0, Packet()
        elif name == "reader":
            if self.foveate:
                lines, boxes = self.reader.read_regions(frame, pre=pre) if pre \
                    else self.reader.read_regions(frame)
            else:
                lines, boxes = self.reader.read(frame), []
            if self.speak_sure and question is None:
                # what OCR is sure of is spoken now; the expert reads only the rest
                said, lines = self.speak_sure(lines)
                if said:
                    yield Say("text", said, "ocr")
                if not lines:
                    return
            if boxes:
                # the reader expert is trained on this: only the text regions,
                # at readable print size, with OCR's words for those regions
                frame, lines = self.foveate(frame, lines, boxes)
                side = 0            # a mosaic is already sized to its text
            if sum(ln.legible for ln in lines) < HINT_MIN_LEGIBLE * len(lines):
                lines = []          # a mostly-"[unreadable]" hint makes the reader refuse
            pk = Packet(text=tuple(lines))
        # vision outranks GPS: when the camera says indoors and the map says
        # street (or the reverse), the place line would contradict the frame
        gps_outdoor = pk.situation.suggested_mode == "outdoor_nav"
        if pk.situation.source != "none" and gps_outdoor != (name == "outdoor") \
                and name != "reader":
            pk = replace(pk, situation=Situation())
        self._described_t, self._described_sig = self.now(), self._signature()
        ex = EXPERTS[name]
        sense = pk.to_prompt(name, self.clock_face)
        self._warmed, self._warm_t = name, self.now()      # this request leaves the cache on `name`
        for chunk in self.llm.stream(name, self.encode(frame, side), sense, question):
            yield Say("describe", chunk, f"expert:{name}")


def demo() -> None:
    from .packet import Obj, TextLine

    t = [0.0]
    calls: list = []

    class G:
        gate = Gate()
        def run(self, f): return self.gate

    class R:
        say, objs = None, []
        def step(self, f, t, s): calls.append("reflex"); return self.objs, self.say

    class Rd:
        lines: list = []
        def read(self, f): calls.append("ocr"); return self.lines

    class P:
        def plan(self, fix, route=None):
            return Situation("street_crossing", "outdoor_nav", source="offline_rules")

    class L:
        def stream(self, name, jpeg, sense, q):
            calls.append(("llm", name, sense, jpeg))
            yield "ok"

    def short(lines):       # (spoken now, lines left for the expert)
        if len(lines) == 1:
            return "Sign says: PUSH.", []
        return ("EXIT.", lines[1:]) if lines[0].text == "EXIT" else (None, lines)

    warmed: list = []
    g, r, rd = G(), R(), Rd()
    arb = Arbiter(now=lambda: t[0])
    c = Conductor(arb, g, r, rd, P(), L(), now=lambda: t[0], speak_sure=short,
                  warm=lambda n, j=None: warmed.append(n), encode=lambda f, s: b"")
    c._spawn = lambda fn, *a: fn(*a)        # warm-ups inline, so the checks below are ordered

    # 1. a blurry frame stops at the gate; with nothing usable, describe gives a cue
    g.gate = Gate(usable=False, issue="blur")
    assert c.on_frame("f0") == [] and calls == []
    assert [s.source for s in c.describe()] == ["gate"]

    # 2. a hazard is spoken straight from the reflex, in whatever mode
    g.gate = Gate(indoor_p=0.1, sharpness=0.9)
    r.say, r.objs = "Pole, 1 o'clock, two meters.", [Obj("pole", 25, dist_m=2, in_path=True)]
    assert c.on_frame("f1") == [Say("hazard", "Pole, 1 o'clock, two meters.", "reflex")]
    r.say = None

    # 3. the planner's situation reaches the arbiter and the expert's prompt
    c.on_fix({"lat": 0, "lon": 0})
    arb.user_command("outdoor_nav")
    c.on_frame("f2")
    out = list(c.describe())
    kind, name, sense, _ = calls[-1]
    assert out[0].source == "expert:outdoor" and name == "outdoor"
    assert "pole right" in sense and "street crossing [GPS]" in sense

    # 4. steady scene: silent. A new thing in the way, after the gap: speak.
    assert not c.scene_changed()
    t[0] += 10
    assert not c.scene_changed()
    r.objs = [Obj("pole", 25, dist_m=2, in_path=True), Obj("person", 0, dist_m=3, in_path=True)]
    c.on_frame("f3")
    assert c.scene_changed()

    # 5. describe mode has no expert of its own: the gate's indoor vote picks one
    arb.user_command("describe")
    g.gate = Gate(indoor_p=0.9)
    c.on_frame("f4")
    assert c.current_expert() == "indoor"

    out = list(c.describe())
    # the planner said street, the camera says indoors: no place line is sent
    assert calls[-1][1] == "indoor" and "GPS" not in calls[-1][2], calls[-1]

    # 6. reading: short text is spoken from OCR and never reaches the model
    arb.user_command("text_reading")
    rd.lines = [TextLine("PUSH", 0.99)]
    n_llm = sum(1 for x in calls if x[0] == "llm")
    assert [s.source for s in c.describe()] == ["ocr"]
    assert sum(1 for x in calls if x[0] == "llm") == n_llm
    # long text goes to the reader expert, with the words and without GPS
    rd.lines = [TextLine("OPEN", 0.99), TextLine("zz", 0.2), TextLine("9 TO 5", 0.995)]
    assert [s.source for s in c.describe()] == ["expert:reader"]
    sense = calls[-1][2]
    assert '"OPEN" | [unreadable] | "9 TO 5"' in sense and "GPS" not in sense

    # a sure sign beside unreadable text: OCR speaks first, the expert gets only the rest
    rd.lines = [TextLine("EXIT", 0.999), TextLine("zz", 0.2, group=1)]
    assert [s.source for s in c.describe()] == ["ocr", "expert:reader"]
    assert "EXIT" not in calls[-1][2]

    # with foveation the model gets the mosaic and only the kept regions' words
    rd.read_regions = lambda f: ([TextLine("OPEN", 0.99, group=0), TextLine("MENU", 0.99, group=1)],
                                 [(0, 0, 9, 9), (0, 20, 9, 29)])
    seen: list = []
    c.foveate, c.encode = (lambda f, ls, b: ("mosaic", ls[1:])), (lambda f, s: seen.append(f) or b"")
    assert [s.source for s in c.describe("What does it say?")] == ["expert:reader"]
    assert seen == ["mosaic"] and '"MENU"' in calls[-1][2] and "OPEN" not in calls[-1][2]
    c.foveate, c.encode = None, (lambda f, s: b"")

    # a frame from before the head turned is not described: stale buffer -> cue
    t[0] += 5
    assert [s.source for s in c.describe()] == ["gate"]

    # 7. the cache follows the expert in use: nothing is warmed for a candidate
    # still in its dwell; the new expert is warmed once, on the frame after the switch
    class Dwelling:                    # an arbiter frozen mid-dwell
        mode, _candidate, motion = "outdoor_nav", "text_reading", Motion.STATIONARY
        def update_gate(self, *a): pass

    c.arb = Dwelling()
    warmed.clear()
    t[0] += 5
    c._warmed = "outdoor"
    g.gate = Gate(indoor_p=0.1)
    c.on_frame("f5"); c.on_frame("f6")
    assert warmed == [], warmed
    Dwelling.mode, Dwelling._candidate = "text_reading", None
    c.on_frame("f5"); c.on_frame("f6")
    assert warmed == ["reader"], warmed

    # 8. a page of text goes to the reader on line detection alone: its
    # mosaic, no OCR recognition, no hints
    rd.layout = lambda f: "pre"
    c.doc_sheet = lambda f, pre: "sheet"
    c.encode = lambda f, s: f.encode() if isinstance(f, str) else b""
    c.arb = arb
    arb.user_command("text_reading")
    c.on_frame("f7")
    n_ocr = calls.count("ocr")
    assert [s.source for s in c.describe()] == ["expert:document"]
    assert calls[-1][3] == b"sheet" and calls[-1][2] == "" and calls.count("ocr") == n_ocr
    print("conductor ok")


if __name__ == "__main__":
    demo()
