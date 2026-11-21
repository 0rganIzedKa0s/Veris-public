"""Local VLM wrapper: Qwen2.5-VL with 4-bit quantization.

The dev box (RTX 3070, 8 GB) has the same memory budget as the Jetson Orin
Nano 8 GB, so a model that fits here is an honest stand-in. On the Jetson the
same weights would run via llama.cpp / TensorRT-LLM instead of transformers,
but prompts, image sizing, and token budgets carry over unchanged.
"""

import queue
import threading
import time
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Cap vision tokens: 4K frames would blow VRAM and latency for no accuracy
# gain at scene-description granularity. ~1.3 MP keeps the encoder under
# ~1700 visual tokens per frame.
MAX_PIXELS = 1280 * 28 * 28 * 1  # qwen-vl-utils convention: multiples of 28x28 patches
MIN_PIXELS = 256 * 28 * 28

# Max seconds the consumer waits for the next streamed chunk. Without a
# timeout, TextIteratorStreamer blocks forever on queue.get() if generate()
# dies before producing a token (e.g. CUDA OOM during prefill).
STREAM_CHUNK_TIMEOUT_S = 300.0


def _stream_from_thread(generate_fn, streamer, gen_kwargs):
    """Run generate_fn(**gen_kwargs) in a daemon thread; yield from streamer.

    generate() only calls streamer.end() on the success path, so an exception
    in the worker thread would otherwise leave the streamer's queue empty
    forever and hang the consumer. Here the worker captures any exception,
    unblocks the streamer, and the exception is re-raised in the consuming
    thread once the stream drains. If the worker produces nothing at all for
    the streamer's timeout window (hung GPU, not an exception), the streamer's
    queue.Empty is surfaced as TimeoutError instead of propagating raw.
    """
    raised: list[BaseException] = []

    def _worker():
        try:
            generate_fn(**gen_kwargs)
        except BaseException as e:  # re-raised below in the consuming thread
            raised.append(e)
            streamer.end()  # enqueue stop signal so the consumer wakes up now

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield from streamer
    except queue.Empty:
        timeout = getattr(streamer, "timeout", None)
        raise TimeoutError(
            f"generation produced no output within {timeout}s (streamer timeout)"
        ) from None
    thread.join(timeout=STREAM_CHUNK_TIMEOUT_S)
    if raised:
        raise raised[0]


@dataclass
class GenStats:
    ttft_s: float          # time to first token — what the user perceives as lag
    total_s: float
    new_tokens: int

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.total_s if self.total_s else 0.0


class VLM:
    def __init__(self, model_id: str = DEFAULT_MODEL, quantize: bool = True):
        kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

    def generate_stream(self, messages, max_new_tokens: int, temperature: float = 0.0):
        """Yield description text chunks as they are generated.

        Streaming is load-bearing for the product: TTS on the device should
        start speaking at the first clause, not after full generation.
        Returns GenStats via StopIteration value (use `yield from` or capture).
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=STREAM_CHUNK_TIMEOUT_S,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        start = time.perf_counter()
        first_token_at: list[float] = []
        n_chunks = 0
        for chunk in _stream_from_thread(self.model.generate, streamer, gen_kwargs):
            if not first_token_at:
                first_token_at.append(time.perf_counter())
            n_chunks += 1
            yield chunk
        end = time.perf_counter()
        self.last_stats = GenStats(
            ttft_s=(first_token_at[0] - start) if first_token_at else end - start,
            total_s=end - start,
            new_tokens=n_chunks,  # chunk count approximates token count with this streamer
        )
