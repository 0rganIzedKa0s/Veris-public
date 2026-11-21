"""The multi-model pipeline behind /api/describe (VERIS_PIPELINE=multi).

One photo from a visitor goes through the code the board benchmark measures
(veris/conductor.py): gate, then detector, then the reader or an expert. Each
request gets its own Conductor, so one visitor's mode and tracks never reach
the next visitor; the models behind it are loaded once.

A photo is not a camera stream. The tracker speaks about an object only after
three frames, so the photo is shown to it three times: a hazard in the photo
is spoken, an approach speed cannot be.
"""

from __future__ import annotations

import json
from dataclasses import replace

MAX_SIDE = 1280     # the wearable frame every tier was measured on


class Pipeline:
    def __init__(self, llm_url: str):
        from bench.orin import e2e      # detector and TensorRT wrappers, as benchmarked
        from veris import reader as reader_mod
        from veris.experts import LlamaClient
        from veris.gate import Gater

        classes = json.loads((e2e.E2E / "scenes.json").read_text())["classes"]
        self.detect, det = e2e.make_detector(classes)
        self.gate, gate = Gater(e2e.MODELS / "gate.onnx", threads=4), "onnxruntime-cpu"
        if (e2e.MODELS / "gate.fp16.engine").exists() and e2e.HAS_TRT:
            self.gate.sess, gate = e2e.TrtSession(e2e.MODELS / "gate.fp16.engine"), "tensorrt-fp16"
        self.reader = reader_mod.Reader(model_dir=e2e.MODELS)
        self.llm = LlamaClient(llm_url, timeout=300)
        self.backends = {"gate": gate, "detector": det, "ocr": "onnxruntime-cpu"}

    def run(self, jpeg: bytes, mode: str, question: str = "", context: str = ""):
        """Yield one trace dict (what the fast tiers saw), then (text, source)
        as each tier speaks."""
        import cv2
        import numpy as np

        from veris import reader as reader_mod
        from veris.arbiter import MODES, Arbiter
        from veris.conductor import Conductor
        from veris.packet import PACKET_MIN_CONF, Situation
        from veris.ranging import Camera
        from veris.reflex import Reflex

        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t = [0.0]
        arb = Arbiter(now=lambda: t[0])
        c = Conductor(arb, self.gate, Reflex(self.detect, Camera.from_hfov(bgr.shape[1], bgr.shape[0])),
                      self.reader, None, self.llm, now=lambda: t[0],
                      speak_sure=reader_mod.speak_sure, foveate=reader_mod.mosaic,
                      doc_sheet=reader_mod.document_sheet)
        arb.user_command(mode if mode in MODES else "describe")
        if context:
            c.situation = Situation(situation=context, source="cloud")
        said = []
        for t[0] in (0.0, 0.1, 0.2):
            said += c.on_frame(bgr)
        g = c.last_gate
        name = c.current_expert() if g.usable else None
        if context and name:    # the visitor typed the place: it is never dropped as a GPS mismatch
            c.packet = replace(c.packet, situation=replace(
                c.situation, suggested_mode="outdoor_nav" if name == "outdoor" else "indoor_nav"))
        yield {"gate": {"usable": g.usable, "issue": g.issue, "indoor_p": round(g.indoor_p, 2),
                        "text_p": round(g.text_p, 2), "sharpness": round(g.sharpness, 2)},
               # what the expert is told: the objects the detector is sure of
               "objects": [o.phrase(False) for o in c.packet.ranked_objects(len(c.packet.objects))
                           if o.conf >= PACKET_MIN_CONF][:6],
               "expert": name, "backends": self.backends}
        for s in said:
            yield s.text, s.source
        for s in c.describe(question or None):
            yield s.text, s.source


def demo() -> None:
    """The wiring, on the dev box: real gate, detector and OCR on CPU, a stub
    in place of the language model."""
    import cv2

    from bench.orin import e2e

    p = Pipeline("http://unused")
    calls = []
    p.llm = type("L", (), {"stream": lambda self, name, jpeg, sense="", q=None: (
        calls.append((name, sense, q)) or iter(["ok"]))})()
    img = cv2.imread(str(sorted(e2e.E2E.glob("street_*.jpg"))[0]))
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    trace, *says = list(p.run(jpeg, "outdoor_nav", context="a street corner"))
    assert trace["gate"]["usable"] and trace["expert"] == "outdoor", trace
    assert says[-1] == ("ok", "expert:outdoor") and "place: a street corner" in calls[-1][1], calls
    blur = cv2.imencode(".jpg", cv2.GaussianBlur(img, (0, 0), 25))[1].tobytes()
    trace, *says = list(p.run(blur, "describe"))
    assert not trace["gate"]["usable"] and says[0][1] == "gate" and len(calls) == 1, (trace, says)
    doc = cv2.imencode(".jpg", cv2.imread(str(e2e.E2E / "doc_0.jpg")))[1].tobytes()
    assert list(p.run(doc, "text_reading"))[-1][1] == "expert:document"
    print("pipeline ok:", trace["backends"])


if __name__ == "__main__":
    demo()
