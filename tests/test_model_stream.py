"""Regression tests for VLM.generate_stream hang-on-error behavior.

If model.generate raises before producing a token (CUDA OOM during prefill
being the prime candidate), the TextIteratorStreamer never receives its stop
signal and the consumer used to block forever on queue.get(). These tests use
stub model/processor objects — no weights are loaded and the GPU is never
touched (run with CUDA_VISIBLE_DEVICES="" to be certain).

  CUDA_VISIBLE_DEVICES="" uv run python -m pytest tests/test_model_stream.py -q
"""

import threading
import time

import pytest
from transformers import TextIteratorStreamer

from veris.model import VLM, _stream_from_thread


class _StubTokenizer:
    """TextIteratorStreamer only stores the tokenizer; it is not called
    unless raw token ids are pushed, which these tests never do."""


class _StubInputs(dict):
    def to(self, device):
        return self


class _StubProcessor:
    tokenizer = _StubTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "stub prompt"

    def __call__(self, text=None, images=None, return_tensors=None):
        return _StubInputs()


class _OOMModel:
    device = "cpu"

    def generate(self, **kwargs):
        raise RuntimeError("CUDA out of memory")


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]


def _make_vlm(model):
    vlm = VLM.__new__(VLM)  # bypass __init__: no weights, no GPU
    vlm.model = model
    vlm.processor = _StubProcessor()
    return vlm


def test_generate_stream_raises_instead_of_hanging_when_generate_fails():
    """generate() dying before the first token must surface as an exception
    in the consuming thread within seconds, not a silent forever-hang."""
    vlm = _make_vlm(_OOMModel())
    result: dict = {}

    def consume():
        try:
            result["chunks"] = list(vlm.generate_stream(MESSAGES, max_new_tokens=8))
        except BaseException as e:  # noqa: BLE001 - recorded for assertions
            result["exc"] = e

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=5)

    assert not consumer.is_alive(), "generate_stream hung instead of raising"
    exc = result.get("exc")
    assert isinstance(exc, RuntimeError)
    assert "CUDA out of memory" in str(exc)


def test_stream_from_thread_success_path_yields_all_chunks():
    streamer = TextIteratorStreamer(_StubTokenizer(), timeout=5)

    def fake_generate(**kwargs):
        # Mimic a successful generate(): emit text, then the stop signal.
        streamer.on_finalized_text("hello ")
        streamer.on_finalized_text("world", stream_end=True)

    chunks = list(_stream_from_thread(fake_generate, streamer, {}))
    assert chunks == ["hello ", "world"]


def test_stream_from_thread_times_out_when_generate_produces_nothing():
    """A worker that hangs without raising (stuck GPU) must trip the streamer
    timeout and surface as TimeoutError, not block forever."""
    streamer = TextIteratorStreamer(_StubTokenizer(), timeout=0.2)
    release = threading.Event()

    def hung_generate(**kwargs):
        release.wait(timeout=10)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        list(_stream_from_thread(hung_generate, streamer, {}))
    assert time.monotonic() - start < 5
    release.set()  # let the daemon worker exit promptly
