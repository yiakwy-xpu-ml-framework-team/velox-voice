"""Multi-kernel decode step driver for the SheetSage2 music abc/scores transcription.

One decode step = embed + 6 x {K3, K1, K2} + lm_head, captured in a single
CUDA graph and replayed per token from Python:

    K1(l): cross-attn
    
    K2(l): cross out GEMV -> res+LN -> ffn1 -> gelu -> ffn2 -> res+LN
    
    K3(l): qkv GEMV -> KV-store -> self-attn -> out GEMV -> res+LN -> xq GEMV
           (multi-SM phase kernels in hopper/block/sheetsage_ar_block.h)

    lm_head: allowed-masked multi-SM scan -> partial argmax (host combine)

    FSM: grammar state update on host, next allowed row + token fed back
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from veloxvoice.kernels.ops.music.cross_attn import cross_attn_decode
from veloxvoice.kernels.ops.music.cuda_ops import (
    ss_gemv, ss_gemv_ln, ss_gemv_row, ss_kvstore, ss_ln_res, ss_self_attn,
    ss_self_attn_split, ss_lmhead)

D = 64
HEADS = 8
MODEL = 512
QKV = 3 * MODEL
FFN = 2048
LAYER_ELEMS = (QKV * MODEL + QKV + MODEL * MODEL + MODEL + MODEL * MODEL + MODEL
               + MODEL * MODEL + MODEL
               + FFN * MODEL + FFN + MODEL * FFN + MODEL + 6 * MODEL)


class MultiKernelStepDecoder:
    """Runs the 6-layer decode step as one captured CUDA graph."""

    def __init__(self, model, k_cross, v_cross, k_self, v_self,
                 w_blob, allowed_table, state_map, n_vocab, eos):
        self.model = model
        dec = model.decoder
        self.k_cross, self.v_cross = k_cross, v_cross
        self.k_self, self.v_self = k_self, v_self
        self.w_blob = w_blob
        self.allowed_table = allowed_table       # [n_states, V] uint8
        self.state_map = state_map               # [n_states, V] int32
        self.embed_w = dec.embed_tokens.weight.detach()
        self.pos_table = dec.embed_positions.weight.detach()
        self.ln_emb_w = dec.layernorm_embedding.weight.detach()
        self.ln_emb_b = dec.layernorm_embedding.bias.detach()
        self.lm_head = model.output_projection.weight.detach()
        self.V, self.eos = n_vocab, eos
        self.S, self.L = k_self.shape[2], k_self.shape[0]
        self.scale = D ** -0.5

        dev = w_blob.device
        self.h = torch.zeros(MODEL, dtype=torch.bfloat16, device=dev)
        self.qkv_buf = torch.zeros(QKV, dtype=torch.bfloat16, device=dev)
        self.tmp = torch.zeros(MODEL, dtype=torch.bfloat16, device=dev)
        self.ffn_buf = torch.zeros(FFN, dtype=torch.bfloat16, device=dev)
        self.s_attn = torch.zeros(HEADS * D, dtype=torch.bfloat16, device=dev)
        self.q_layer = torch.zeros(HEADS * D, dtype=torch.bfloat16, device=dev)
        self.a_layer = torch.zeros(HEADS * D, dtype=torch.bfloat16, device=dev)
        # device-side step inputs
        self.pos_dev = torch.zeros(1, dtype=torch.int32, device=dev)
        self.x_in = torch.zeros(1, dtype=torch.long, device=dev)
        self.allowed_cur = torch.zeros(self.V, dtype=torch.uint8, device=dev)
        self.attn_split = 16
        self.spm = torch.zeros(HEADS, self.attn_split, dtype=torch.float32, device=dev)
        self.spl = torch.zeros(HEADS, self.attn_split, dtype=torch.float32, device=dev)
        self.spacc = torch.zeros(HEADS, self.attn_split, D, dtype=torch.float32, device=dev)
        self.lm_part_b = torch.zeros(512, dtype=torch.float32, device=dev)
        self.lm_part_i = torch.zeros(512, dtype=torch.int32, device=dev)
        self._graph = None
        self._lviews = [self._views(l) for l in range(self.L)]

    def _views(self, l):
        o = l * LAYER_ELEMS
        w = self.w_blob

        def v(off, n):
            return w[o + off: o + off + n]
        p = QKV * MODEL
        qkv_w = v(0, p).view(QKV, MODEL)
        qkv_b = v(p, QKV); p += QKV
        out_w = v(p, MODEL * MODEL).view(MODEL, MODEL); p += MODEL * MODEL
        out_b = v(p, MODEL); p += MODEL
        xq_w = v(p, MODEL * MODEL).view(MODEL, MODEL); p += MODEL * MODEL
        xq_b = v(p, MODEL); p += MODEL
        xo_w = v(p, MODEL * MODEL).view(MODEL, MODEL); p += MODEL * MODEL
        xo_b = v(p, MODEL); p += MODEL
        f1_w = v(p, FFN * MODEL).view(FFN, MODEL); p += FFN * MODEL
        f1_b = v(p, FFN); p += FFN
        f2_w = v(p, MODEL * FFN).view(MODEL, FFN); p += MODEL * FFN
        f2_b = v(p, MODEL); p += MODEL
        lns = v(p, 6 * MODEL)
        return dict(qkv_w=qkv_w, qkv_b=qkv_b, out_w=out_w, out_b=out_b,
                    xq_w=xq_w, xq_b=xq_b, xo_w=xo_w, xo_b=xo_b,
                    f1_w=f1_w, f1_b=f1_b, f2_w=f2_w,
                    f2_b=f2_b, lns=lns)

    def _embed(self):
        e = F.embedding(self.x_in, self.embed_w)               # [1, 512]
        pe = F.embedding(self.pos_dev.long() + 2, self.pos_table)  # [1, 512]
        self.h.copy_(e[0] + pe[0])
        ss_ln_res(self.h, None, self.ln_emb_w, self.ln_emb_b)

    def _layer(self, l):
        """K3 + K1 + K2 with every residual+LayerNorm folded into the next
        gemv's prologue (no standalone LN kernels; gelu folded into ffn1)."""
        v = self._lviews[l]
        h, qb, tmp, ffn = self.h, self.qkv_buf, self.tmp, self.ffn_buf
        lns = v["lns"]
        e = 1e-5
        
        # K3 (residual+LayerNorm folded into the following gemv prologue)
        if l == 0:
            ss_gemv(v["qkv_w"], v["qkv_b"], h, qb, QKV, MODEL, 256)
        else:
            pv = self._lviews[l - 1]["lns"]
            ss_gemv_ln(v["qkv_w"], v["qkv_b"], h, tmp,
                       pv[4 * MODEL:5 * MODEL], pv[5 * MODEL:6 * MODEL],
                       qb, QKV, MODEL, 1e-5, False, 256)
        ss_kvstore(qb, self.k_self[l], self.v_self[l], self.S, self.pos_dev)
        ss_self_attn_split(qb, self.k_self[l], self.v_self[l], self.s_attn,
                           self.spm, self.spl, self.spacc, self.S,
                           self.pos_dev, self.attn_split, HEADS, self.scale)
        ss_gemv(v["out_w"], v["out_b"], self.s_attn, tmp, MODEL, MODEL, 64)
        
        # K1
        ss_gemv_ln(v["xq_w"], v["xq_b"], h, tmp, lns[0:MODEL],
                   lns[MODEL:2 * MODEL], self.q_layer, MODEL, MODEL,
                   1e-5, False, 64)
        self.a_layer.copy_(cross_attn_decode(
            self.q_layer.view(HEADS, D), self.k_cross, self.v_cross, l).reshape(-1))
        
        # K2
        ss_gemv(v["xo_w"], v["xo_b"], self.a_layer, tmp, MODEL, MODEL, 64)
        ss_gemv_ln(v["f1_w"], v["f1_b"], h, tmp, lns[2 * MODEL:3 * MODEL],
                   lns[3 * MODEL:4 * MODEL], ffn, FFN, MODEL, 1e-5, True, 256)
        ss_gemv_row(v["f2_w"], v["f2_b"], ffn, tmp, MODEL, FFN)

    def _final_ln(self):
        lns = self._lviews[self.L - 1]["lns"]
        ss_ln_res(self.h, self.tmp, lns[4 * MODEL:5 * MODEL],
                  lns[5 * MODEL:6 * MODEL])

    def _lm_head(self):
        ss_lmhead(self.lm_head, self.h, self.allowed_cur,
                  self.lm_part_b, self.lm_part_i, self.V, MODEL, 512)

    def _body(self):
        self._embed()
        for l in range(self.L):
            self._layer(l)
        self._final_ln()
        self._lm_head()

    def capture(self):
        # capture on the current stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._body()
        torch.cuda.synchronize()

    def next_token(self) -> int:
        """Replay the grape for the current (pos, token, allowed) state."""
        self._graph.replay()
        b = int(self.lm_part_b.argmax())
        return int(self.lm_part_i[b])


_DECODER_CACHE: dict = {}


def pack_decoder_weights(dec):
    """Flatten one decoder layer's weights into the single contiguous bf16 blob"""
    parts = []
    for layer in dec.layers:
        qkv_w = torch.cat([layer.self_attn.q_proj.weight,
                           layer.self_attn.k_proj.weight,
                           layer.self_attn.v_proj.weight], 0)
        qkv_b = torch.cat([layer.self_attn.q_proj.bias,
                           layer.self_attn.k_proj.bias,
                           layer.self_attn.v_proj.bias], 0)
        parts += [qkv_w.reshape(-1), qkv_b.reshape(-1),
                  layer.self_attn.out_proj.weight.reshape(-1),
                  layer.self_attn.out_proj.bias.reshape(-1),
                  layer.encoder_attn.q_proj.weight.reshape(-1),
                  layer.encoder_attn.q_proj.bias.reshape(-1),
                  layer.encoder_attn.out_proj.weight.reshape(-1),
                  layer.encoder_attn.out_proj.bias.reshape(-1),
                  layer.fc1.weight.reshape(-1), layer.fc1.bias.reshape(-1),
                  layer.fc2.weight.reshape(-1), layer.fc2.bias.reshape(-1),
                  layer.self_attn_layer_norm.weight.reshape(-1),
                  layer.self_attn_layer_norm.bias.reshape(-1),
                  layer.encoder_attn_layer_norm.weight.reshape(-1),
                  layer.encoder_attn_layer_norm.bias.reshape(-1),
                  layer.final_layer_norm.weight.reshape(-1),
                  layer.final_layer_norm.bias.reshape(-1)]
    parts += [dec.layernorm_embedding.weight.reshape(-1),
              dec.layernorm_embedding.bias.reshape(-1)]
    return torch.cat(parts).contiguous()


def _get_decoder(model, imp, w_blob, at, sm):
    """Cached MultiKernelStepDecoder + captured graph, keyed by the model and
    the _GraphedStep's KV buffers"""
    key = (id(model), imp.k_self.data_ptr(), imp.S, imp.T)
    deco = _DECODER_CACHE.get(key)

    if deco is None:
        deco = MultiKernelStepDecoder(
            model, imp.k_cross.view(6, 8, imp.T, 64),
            imp.v_cross.view(6, 8, imp.T, 64),
            imp.k_self.view(6, 8, imp.S, 64), imp.v_self.view(6, 8, imp.S, 64),
            w_blob, at, sm, model.config.vocab_size, model.tokenizer.eos_token)

        deco.capture()

        _DECODER_CACHE.clear()
        _DECODER_CACHE[key] = deco
    return deco


def transcribe_window_multi(model, audio, prompts, max_sequence_length,
                            *, memory=None, stop_time_seconds=None,
                            prefix_tokens=None):
    """Whole-window decode on the multi-kernel CUDA-graph pipeline.

    Drop-in for ``grammar_ar_decode.constrained_prompt_generate`` (same
    prefix/grammar contract); returns a ``torch.long`` token tensor.
    """
    from veloxvoice.stream.grammar_ar_decode import (
        build_grammar_tables, PromptGrammarState)

    tok = model.tokenizer
    prefix = (list(prefix_tokens) if prefix_tokens is not None
              else tok.prompt_prefix(tok.normalize_prompts(prompts)))
    if prefix[-1] == tok.eos_token:
        prefix = prefix[:-1]
    g = getattr(model, "_graphed", None)
    if g is None:
        raise ValueError("multi-kernel pipeline needs the graphed decoder "
                         "(host prefill shares its _GraphedStep caches)")
    if memory is None:
        memory = model.encode(audio)
    memory = memory.contiguous()
    if memory.dim() == 2:
        memory = memory[None]

    with torch.inference_mode():
        imp = g._ensure(memory)
        imp.prefill_full_sequence(
            torch.tensor([prefix], dtype=torch.long, device=memory.device))
        imp.refresh_cross(memory)

        w_blob = pack_decoder_weights(model.decoder)

        at, sm, n_states = build_grammar_tables(tok, prompts, str(memory.device))
        deco = _get_decoder(model, imp, w_blob, at, sm)

        state = PromptGrammarState(tok)
        out_ix = prefix.index(tok.out_token)
        for t in prefix[out_ix + 1:]:
            state.update(t)
        # device FSM state after the (possibly non-empty) prefix
        state_id = 0
        for t in prefix[out_ix + 1:]:
            state_id = int(sm[state_id][t])
        token = prefix[-1]
        pos = len(prefix)
        out = list(prefix)
        for _ in range(int(max_sequence_length)):
            deco.pos_dev.fill_(pos)
            deco.x_in.fill_(token)
            deco.allowed_cur.copy_(deco.allowed_table[state_id])
            token = deco.next_token()
            out.append(token)
            done = state.update(token)
            if done or token == tok.eos_token:
                break
            stop = (stop_time_seconds is not None
                    and tok.time_token_start <= token < tok.time_token_end
                    and tok.token_to_time_id(token) / tok.time_hz
                    >= stop_time_seconds)
            if stop:
                out.append(tok.eos_token)
                break
            state_id = int(sm[state_id][token])
            pos += 1
    if out[-1] != tok.eos_token:
        out.append(tok.eos_token)
    return torch.tensor(out, dtype=torch.long)
