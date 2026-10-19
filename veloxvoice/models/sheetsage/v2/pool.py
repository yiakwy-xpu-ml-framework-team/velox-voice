"""Window decode pool for the SheetSage2 serving"""
from __future__ import annotations

import os
import threading
import time

import torch


class _Slot:
    __slots__ = ("memory", "prompts", "max_len", "stop", "event",
                 "tokens", "error")

    def __init__(self, memory, prompts, max_len, stop):
        self.memory = memory
        self.prompts = prompts
        self.max_len = max_len
        self.stop = stop
        self.event = threading.Event()
        self.tokens = None
        self.error = None


class WindowDecodePool:
    """Serialize decoder entry; batch prefix-free windows across requests."""

    def __init__(self, model):
        self.model = model
        graphed = getattr(model, "_graphed", None)
        self.enabled = (graphed is not None
                        and graphed.backend == "piecewise_graph")
        self.max_batch = graphed.batches[-1] if graphed is not None else 1
        self.batch_window = float(
            os.environ.get("VELOXVOICE_MUSIC_BATCH_WINDOW_MS", "120")) / 1000.0
        self._qlock = threading.Lock()          # cohort bookkeeping only
        self._cohort = threading.Condition(self._qlock)
        self._decode_lock = threading.Lock()    # held for whole decodes
        self._waiting: list[_Slot] = []
        self._active = 0                        # in-flight transcribe requests

    def transcribe_window(self, segment, prompts, max_seq_len, *,
                          prefix_tokens, autocast_dtype, stop_time_seconds,
                          memory):
        """Decode one window; mirrors ``_generate_window``'s contract and
        returns a ``torch.long`` tensor of token ids."""
        self._active += 1
        try:
            return self._transcribe_window(
                segment, prompts, max_seq_len,
                prefix_tokens=prefix_tokens, autocast_dtype=autocast_dtype,
                stop_time_seconds=stop_time_seconds, memory=memory)
        finally:
            self._active -= 1

    def _transcribe_window(self, segment, prompts, max_seq_len, *,
                           prefix_tokens, autocast_dtype, stop_time_seconds,
                           memory):
        from ....stream.grammar_ar_decode import constrained_prompt_generate

        from .api import _decode_engine

        engine = _decode_engine()
        pipeline = engine == "pipeline"
        if not self.enabled or prefix_tokens is not None or pipeline:
            with self._decode_lock:
                if pipeline and getattr(self.model, "_graphed", None) is not None:
                    try:
                        from .step_pipeline import transcribe_window_multi

                        return transcribe_window_multi(
                            self.model, segment, prompts, max_seq_len,
                            prefix_tokens=prefix_tokens,
                            stop_time_seconds=stop_time_seconds, memory=memory)
                    except Exception:
                        pass
                return constrained_prompt_generate(
                    self.model, segment, prompts, max_seq_len,
                    prefix_tokens=prefix_tokens,
                    autocast_dtype=autocast_dtype,
                    stop_time_seconds=stop_time_seconds, memory=memory)

        slot = _Slot(memory, prompts, max_seq_len, stop_time_seconds)
        with self._cohort:
            first = not self._waiting
            self._waiting.append(slot)
            self._cohort.notify_all()

        if first:
            if self._active <= 1:
                self._run_cohort(wait=False)
            else:
                self._run_cohort()
        else:
            if not slot.event.wait(timeout=900):
                slot.error = TimeoutError("decode cohort wait timed out")

        if slot.error is not None:
            raise slot.error
        return torch.tensor(slot.tokens, dtype=torch.long)

    def _run_cohort(self, wait=True):
        deadline = time.monotonic() + self.batch_window
        with self._cohort:
            while wait and len(self._waiting) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cohort.wait(remaining)
            batch = self._waiting[:]
            self._waiting.clear()
        try:
            from ....stream.grammar_ar_decode import batched_constrained_generate

            with self._decode_lock:
                tokens_list = batched_constrained_generate(
                    self.model, [s.memory for s in batch], batch[0].prompts,
                    batch[0].max_len,
                    stop_time_seconds=[s.stop for s in batch])
                    
            for slot, tokens in zip(batch, tokens_list):
                slot.tokens = tokens
        except Exception as exc:  # fan out; never strand a waiter
            for slot in batch:
                slot.error = exc
        finally:
            for slot in batch:
                slot.event.set()
