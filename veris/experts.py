"""Tier 2: one base model, one adapter per job.

The base (Qwen2.5-VL-3B) and its vision tower load once. Each expert is a
LoRA adapter that llama-server keeps loaded and selects per request, so a
switch is a field in the request body, not a reload.

Expert prompts are one sentence. The behaviour the old 350-token prompts
described is trained into the adapter, so the prompt only has to name the
job. A short prompt is what makes a switch cheap: there is almost no prefix
to prefill when the adapter changes.

The planner is the same base with every adapter off, text only.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True)
class Expert:
    name: str
    adapter: str | None     # GGUF adapter name, None = bare base
    system: str
    ask: str                # default user request
    image_side: int         # long-side budget in px; 0 = no image
    max_tokens: int


EXPERTS: dict[str, Expert] = {
    "outdoor": Expert(
        "outdoor", "outdoor",
        "Veris outdoor. Tell the blind user what is around them: the path and "
        "anything in it first, then landmarks. Second person, short sentences, "
        "only what is visible.",
        "What's around me?", 448, 90),
    "indoor": Expert(
        "indoor", "indoor",
        "Veris indoor. Tell the blind user what room this is, then obstacles "
        "and openings, then furniture nearest first. Second person, short "
        "sentences, only what is visible.",
        "What's around me?", 560, 120),
    "reader": Expert(
        "reader", "reader",
        "Veris reader. Read the text to the blind user word for word, most "
        "prominent source first, naming each source. Say where text cannot be "
        "read. Never complete a word.",
        "Read this to me.", 448, 300),
    "planner": Expert(
        "planner", None,
        "You answer a blind pedestrian's question about where things are, using "
        "only the map facts given. One short sentence with the distance and the "
        "side. If the facts do not list it, say it is not on the map nearby.",
        "", 0, 120),
}
# A page, a menu, a receipt: the reader's prompt on the bare base model. The
# reader adapter is trained on signs and gives up on print it finds hard; on
# receipt photos the base reads 74% of lines from the same crop, the adapter
# 60% (evals/results/reader_docs.json).
EXPERTS["document"] = replace(EXPERTS["reader"], name="document", adapter=None)


def expert_for(arbiter_mode: str, indoor_p: float = 0.5) -> str:
    """Arbiter mode (veris/arbiter.py vocabulary) -> expert name. `describe`
    has no expert of its own: the gate's indoor vote picks one."""
    return {
        "outdoor_nav": "outdoor",
        "indoor_nav": "indoor",
        "text_reading": "reader",
    }.get(arbiter_mode, "indoor" if indoor_p >= 0.5 else "outdoor")


def user_text(expert: Expert, packet_text: str = "", question: str | None = None) -> str:
    """The exact user turn the experts are trained on."""
    ask = question or expert.ask
    return f"{ask}\n{packet_text}" if packet_text else ask


@dataclass
class Timing:
    ttft_s: float = 0.0
    total_s: float = 0.0
    tokens: int = 0


class LlamaClient:
    """llama-server client. Adapters are loaded at server start with
    `--lora a.gguf --lora b.gguf --lora-init-without-apply`; each request
    names the one it wants through the per-request `lora` field."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self._ids: dict[str, int] | None = None
        self.last = Timing()

    def adapter_ids(self) -> dict[str, int]:
        if self._ids is None:
            with urllib.request.urlopen(self.base_url + "/lora-adapters",
                                        timeout=self.timeout) as r:
                rows = json.load(r)
            # match by file stem: ".../veris-outdoor.gguf" -> "outdoor"
            self._ids = {}
            for row in rows:
                stem = row["path"].rsplit("/", 1)[-1].removesuffix(".gguf")
                for name in ("outdoor", "indoor", "reader"):
                    if name in stem:
                        self._ids[name] = row["id"]
        return self._ids

    def _lora(self, adapter: str | None) -> list[dict]:
        ids = self.adapter_ids()
        if adapter is not None and adapter not in ids:
            raise KeyError(f"adapter {adapter!r} is not loaded on the server "
                           f"(loaded: {sorted(ids)})")
        return [{"id": i, "scale": 1.0 if n == adapter else 0.0}
                for n, i in ids.items()]

    def stream(self, expert_name: str, jpeg: bytes | None = None,
               packet_text: str = "", question: str | None = None,
               max_tokens: int | None = None) -> Iterator[str]:
        """Yield text chunks. Timing for the call is left in `self.last`."""
        ex = EXPERTS[expert_name]
        content: list[dict] = []
        if jpeg is not None and ex.image_side:
            b64 = base64.b64encode(jpeg).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": user_text(ex, packet_text, question)})
        body = {
            "messages": [{"role": "system", "content": ex.system},
                         {"role": "user", "content": content}],
            "max_tokens": max_tokens or ex.max_tokens,
            "temperature": 0.0, "stream": True, "cache_prompt": True,
            "lora": self._lora(ex.adapter),
        }
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        self.last = Timing()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                if delta:
                    if not self.last.tokens:
                        self.last.ttft_s = time.perf_counter() - t0
                    self.last.tokens += 1
                    yield delta
        self.last.total_s = time.perf_counter() - t0

    def run(self, *a, **kw) -> str:
        return "".join(self.stream(*a, **kw)).strip()

    def warm(self, expert_name: str, jpeg: bytes | None = None) -> None:
        """Prefill an expert's system prefix so its next real request does not
        pay for it. One token. With `jpeg`, the image is encoded and cached
        too, so a request for the same image starts at the text after it."""
        self.run(expert_name, jpeg, max_tokens=1)


def demo() -> None:
    assert expert_for("outdoor_nav") == "outdoor"
    assert expert_for("text_reading") == "reader"
    assert expert_for("describe", indoor_p=0.9) == "indoor"
    assert expert_for("describe", indoor_p=0.1) == "outdoor"
    ex = EXPERTS["outdoor"]
    assert user_text(ex) == "What's around me?"
    assert user_text(ex, "<sense>\nobjects: car\n</sense>", "Is it safe?") == \
        "Is it safe?\n<sense>\nobjects: car\n</sense>"
    # the trained-in prompts must stay short: that is the point of them
    assert all(len(e.system.split()) < 45 for e in EXPERTS.values())
    c = LlamaClient()
    c._ids = {"outdoor": 0, "indoor": 1, "reader": 2}
    assert c._lora("indoor") == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0},
                                 {"id": 2, "scale": 0.0}]
    assert all(x["scale"] == 0.0 for x in c._lora(None))      # planner = bare base
    print("experts ok")


if __name__ == "__main__":
    demo()
