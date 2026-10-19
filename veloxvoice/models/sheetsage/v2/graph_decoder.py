"""CUDA-graphed SheetSage2 Bart decode step"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


class _GraphedStep:

    def __init__(self, decoder, out_proj, vocab: int, s_max: int, t_mem: int,
                 dtype: torch.dtype = torch.bfloat16,
                 backend: str = "mega_decode_step",
                 batch: int = 1):
        self.dec = decoder
        self.out_proj = out_proj
        self.backend = backend
        self.B = max(1, int(batch))
        self.L = len(decoder.layers)
        self.H = decoder.layers[0].self_attn.num_heads
        self.D = decoder.layers[0].self_attn.head_dim
        self.scale = self.D ** -0.5
        self.V = vocab
        self.S = s_max
        self.T = t_mem
        self.dtype = dtype
        dev = decoder.embed_tokens.weight.device
        self.dev = dev

        # static inputs / outputs (per slot)
        self.x_in = torch.zeros(self.B, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.allowed = torch.zeros(self.B, vocab, dtype=torch.bool, device=dev)
        self.logits = torch.zeros(self.B, 1, vocab, dtype=dtype, device=dev)
        self.token_out = torch.zeros(self.B, 1, dtype=torch.long, device=dev)
        self.slots = torch.arange(self.B, device=dev)

        # KV caches ([L, B, H, S, D] / [L, B, H, T, D], bf16 — same dtype as
        # the reference cache; attention reads them directly)
        self.k_self = torch.zeros(self.L, self.B, self.H, s_max, self.D,
                                  dtype=dtype, device=dev)
        self.v_self = torch.zeros_like(self.k_self)
        self.k_cross = torch.zeros(self.L, self.B, self.H, t_mem, self.D,
                                   dtype=dtype, device=dev)
        self.v_cross = torch.zeros_like(self.k_cross)

        # position table rows (learned embedding, offset 2 handled at gather)
        self.pos_table = decoder.embed_positions.weight.detach()
        self.col = torch.arange(s_max, device=dev)
        self.neg = torch.finfo(torch.float32).min

        # pre-packed QKV projections (3 GEMMs -> 1 per attention per step)
        def _pack_qkv(attn):
            w = torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                           attn.v_proj.weight], dim=0).contiguous()
            b = (torch.cat([attn.q_proj.bias, attn.k_proj.bias,
                            attn.v_proj.bias], dim=0).contiguous()
                 if attn.q_proj.bias is not None else None)
            return w, b

        self.qkv_w, self.qkv_b = zip(*[_pack_qkv(l.self_attn)
                                       for l in decoder.layers])
        self.xqkv_w, self.xqkv_b = zip(*[_pack_qkv(l.encoder_attn)
                                         for l in decoder.layers])

        self.graph = None
        self._fused_ready = False
        if backend == "mega_decode_step":
            self._init_fused()

    def _init_fused(self):
        """Pack per-layer weights into one blob + alloc scratch.

        The blob layout is documented in ``velox_sheetsage_step.cu`` and must
        stay in sync with the kernel's constexpr offsets.
        """
        import torch

        dec = self.dec
        H, D = self.H, self.D

        parts = []
        for l in dec.layers:
            qkv_w = torch.cat([l.self_attn.q_proj.weight,
                               l.self_attn.k_proj.weight,
                               l.self_attn.v_proj.weight], dim=0)
            qkv_b = torch.cat([l.self_attn.q_proj.bias,
                               l.self_attn.k_proj.bias,
                               l.self_attn.v_proj.bias], dim=0)
            parts += [qkv_w.reshape(-1), qkv_b.reshape(-1),
                      l.self_attn.out_proj.weight.reshape(-1),
                      l.self_attn.out_proj.bias.reshape(-1),
                      l.encoder_attn.q_proj.weight.reshape(-1),
                      l.encoder_attn.q_proj.bias.reshape(-1),
                      l.fc1.weight.reshape(-1), l.fc1.bias.reshape(-1),
                      l.fc2.weight.reshape(-1), l.fc2.bias.reshape(-1),
                      l.self_attn_layer_norm.weight.reshape(-1),
                      l.self_attn_layer_norm.bias.reshape(-1),
                      l.encoder_attn_layer_norm.weight.reshape(-1),
                      l.encoder_attn_layer_norm.bias.reshape(-1),
                      l.final_layer_norm.weight.reshape(-1),
                      l.final_layer_norm.bias.reshape(-1)]
        parts += [dec.layernorm_embedding.weight.reshape(-1),
                  dec.layernorm_embedding.bias.reshape(-1)]
        self.f_w_blob = torch.cat(parts).contiguous()
        self.f_tok = dec.embed_tokens.weight.detach()          # [V,512]
        self.f_lm = self.out_proj.weight.detach()              # tied [V,512]

        # small scratch + reduction blobs (sized for up to 2048 blocks)
        self.f_scratch = torch.zeros(512 + 1536 + 2048 + 512 + 512,
                                     dtype=self.dtype, device=self.dev)
        self.f_red = torch.zeros(2048 * (H * 2 + H * D + 2),
                                 dtype=torch.float32, device=self.dev)
        self.f_pos32 = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.f_tok_out = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.f_logits = torch.zeros(self.V, dtype=self.dtype, device=self.dev)
        self.f_dbg = None
        if os.environ.get("VELOX_SSDBG"):
            self.f_dbg = torch.zeros(8, dtype=torch.float32, device=self.dev)
        self._fused_ready = True

    def _fused_body(self):
        from veloxvoice.kernels.ops.music.cuda_ops import sheetsage_step

        # [L,1,H,S,D] -> [L,H,S,D] pointer-compatible views
        ks = self.k_self.view(self.L, self.H, self.S, self.D)
        vs = self.v_self.view(self.L, self.H, self.S, self.D)
        kc = self.k_cross.view(self.L, self.H, self.T, self.D)
        vc = self.v_cross.view(self.L, self.H, self.T, self.D)
        self.f_pos32.copy_(self.pos)  # int64 -> int32, device-side
        sheetsage_step(
            self.x_in, self.f_pos32, self.allowed.view(-1),
            self.f_w_blob, self.f_tok, self.pos_table,
            ks, vs, kc, vc, self.f_lm,
            self.f_scratch, self.f_red, self.f_logits, self.f_tok_out,
            self.f_dbg if self.f_dbg is not None else torch.zeros(
                0, dtype=torch.float32, device=self.dev),
            self.S, self.T)
        self.token_out.copy_(self.f_tok_out)
        self.logits.view(-1).copy_(self.f_logits)

    def _body(self):
        return self._fused_body() if self.backend == "mega_decode_step" \
            else self._step_body()

    def _run_body(self):
        if self.backend == "mega_decode_step":
            self._fused_body()  # cooperative launch: no graph capture
        else:
            self.graph.replay()

    @torch.inference_mode()
    def prefill_full_sequence(self, decoder_input_ids):
        dec = self.dec
        prompt = decoder_input_ids
        if prompt.dim() == 1:
            prompt = prompt[None]
        B, P = prompt.shape
        h = dec.embed_tokens(prompt)
        pe = dec.embed_positions.weight[2:P + 2]
        h = h + pe.view(1, P, -1).to(h.dtype)
        h = dec.layernorm_embedding(h)

        causal = torch.triu(
            torch.full((P, P), torch.finfo(self.dtype).min, device=h.device),
            diagonal=1).to(self.dtype)

        k_self_rows, v_self_rows = [], []
        for l, layer in enumerate(dec.layers):
            attn = layer.self_attn
            qkv = F.linear(h, self.qkv_w[l], self.qkv_b[l])
            q, k, v = qkv.split(self.D * self.H, dim=-1)
            q = q.view(B, P, self.H, self.D).transpose(1, 2)
            k = k.view(B, P, self.H, self.D).transpose(1, 2)
            v = v.view(B, P, self.H, self.D).transpose(1, 2)
            k_self_rows.append(k.detach())
            v_self_rows.append(v.detach())
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            a = a.transpose(1, 2).reshape(B, P, -1)
            h = layer.self_attn.out_proj(a) + h
            h = layer.self_attn_layer_norm(h)

            attn = layer.encoder_attn
            q = attn.q_proj(h).view(B, P, self.H, self.D).transpose(1, 2)
            a = F.scaled_dot_product_attention(
                q, self.k_cross[l], self.v_cross[l])
            a = a.transpose(1, 2).reshape(B, P, -1)
            h = attn.out_proj(a) + h
            h = layer.encoder_attn_layer_norm(h)

            h = layer.fc2(F.gelu(layer.fc1(h))) + h
            h = layer.final_layer_norm(h)

        for l in range(self.L):
            self.k_self[l][:, :, :P, :] = k_self_rows[l]
            self.v_self[l][:, :, :P, :] = v_self_rows[l]
        logits = self.out_proj(h[:, -1:])
        return logits

    @torch.inference_mode()
    def refresh_cross(self, memory: torch.Tensor) -> None:
        """(Re)compute cross-attention K/V from a batch of encoder memories.

        ``memory``: [B, T, C] (or [1, T, C] for the single-slot case).
        """
        if memory.dim() == 2:
            memory = memory[None]
        h = memory.to(self.dtype)
        B = h.shape[0]
        for l, layer in enumerate(self.dec.layers):
            k = layer.encoder_attn.k_proj(h)
            v = layer.encoder_attn.v_proj(h)
            k = k.view(B, self.T, self.H, self.D).transpose(1, 2)
            v = v.view(B, self.T, self.H, self.D).transpose(1, 2)
            self.k_cross[l].copy_(k)
            self.v_cross[l].copy_(v)

    @torch.inference_mode()
    def _step_body(self) -> None:
        dec = self.dec
        B = self.B
        pos = self.pos  # [B] long, device
        h = dec.embed_tokens(self.x_in)  # [B,1,D]
        pe = self.pos_table.index_select(0, pos + 2)
        h = h + pe.view(B, 1, -1).to(h.dtype)
        h = dec.layernorm_embedding(h)

        neg = torch.finfo(self.dtype).min
        for l, layer in enumerate(dec.layers):
            qkv = F.linear(h, self.qkv_w[l], self.qkv_b[l])
            q, k, v = qkv.split(self.D * self.H, dim=-1)

            q = q.view(B, 1, self.H, self.D).transpose(1, 2)
            k = k.view(B, 1, self.H, self.D).transpose(1, 2)
            v = v.view(B, 1, self.H, self.D).transpose(1, 2)

            self.k_self[l][self.slots, :, pos, :] = k[:, :, 0, :]
            self.v_self[l][self.slots, :, pos, :] = v[:, :, 0, :]

            scores = torch.matmul(q, self.k_self[l].transpose(-1, -2)) * self.scale
            bias = ((self.col > pos.view(-1, 1)).view(B, 1, 1, -1) * neg).to(self.dtype)
            probs = torch.softmax(scores.float() + bias.float(), dim=-1).to(self.dtype)
            a = torch.matmul(probs, self.v_self[l])
            a = a.transpose(1, 2).reshape(B, 1, -1)
            h = layer.self_attn.out_proj(a) + h
            h = layer.self_attn_layer_norm(h)

            # cross attention (single query per slot, full memory)
            qkv = F.linear(h, self.xqkv_w[l], self.xqkv_b[l])
            q = qkv[..., :self.D * self.H]
            q = q.view(B, 1, self.H, self.D).transpose(1, 2)
            scores = torch.matmul(q, self.k_cross[l].transpose(-1, -2)) * self.scale
            probs = torch.softmax(scores.float(), dim=-1).to(self.dtype)
            a = torch.matmul(probs, self.v_cross[l])
            a = a.transpose(1, 2).reshape(B, 1, -1)
            h = layer.encoder_attn.out_proj(a) + h
            h = layer.encoder_attn_layer_norm(h)

            # ffn (gelu)
            h = layer.fc2(F.gelu(layer.fc1(h))) + h
            h = layer.final_layer_norm(h)

        logits = self.out_proj(h)  # [B,1,V] bf16
        self.logits.copy_(logits)
        masked = logits.float().masked_fill(~self.allowed.view(B, 1, -1), self.neg)
        self.token_out.copy_(masked.argmax(dim=-1))

    @torch.inference_mode()
    def _capture(self) -> None:
        if self.backend == "mega_decode_step":
            self.graph = None
            return

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._body()
        self.graph = g


class _AllowedCache:
    """Cache grammar allowed-masks"""

    def __init__(self, device):
        self.device = device
        self.cache = {}

    def tensor_for(self, state):
        tok = state.tokenizer
        can_end = state.payload_count > 0
        allow_shift = (state.payload_count > 0 or state.in_shift) and state.shift_run < 4
        key = (can_end, allow_shift, state.incomplete, state.last_field_index)
        t = self.cache.get(key)
        if t is None:
            t = state.allowed(self.device)
            self.cache[key] = t
        return t


class GraphedSheetSage2Decoder:
    def __init__(self, model, s_max: int | None = None, t_mem: int = 7500,
                 backend: str | None = None, batches=(1, 2, 4, 8)):
        import os

        self.model = model
        self.dec = model.decoder
        self.vocab = model.config.vocab_size
        self.s_max = s_max or model.max_output_seq_len
        self.t_mem = t_mem
        self.backend = backend or os.environ.get("VELOXVOICE_SHEETSAGE_STEP",
                                                 "piecewise_graph")
        self.batches = tuple(sorted({1} | {int(b) for b in batches if b > 1}))
        self._impls = {}       # B -> _GraphedStep
        self._mem_ptrs = {}    # B -> last cross-KV memory data_ptr
        self.state = None
        self.allowed_cache = None

    def _ensure(self, memory: torch.Tensor) -> _GraphedStep:
        B = 1 if memory.dim() == 2 else memory.shape[0]

        if self.backend == "mega_decode_step":
            B = 1

        bucket = next((b for b in self.batches if b >= B), self.batches[-1])
        imp = self._impls.get(bucket)
        if imp is None:
            imp = _GraphedStep(self.dec, self.model.output_projection,
                               self.vocab, self.s_max, self.t_mem,
                               dtype=memory.dtype,
                               backend=self.backend, batch=bucket)
            imp._capture()
            self._impls[bucket] = imp
            if self.allowed_cache is None:
                self.allowed_cache = _AllowedCache(imp.dev)
        key = (bucket, memory.data_ptr())
        if self._mem_ptrs.get(bucket) != key:
            imp.refresh_cross(memory)
            self._mem_ptrs[bucket] = key
        return imp

    @torch.inference_mode()
    def decode(self, memory, decoder_input_ids, use_cache=True,
               past_key_values=None):
        imp = self._ensure(memory)
        if decoder_input_ids.dim() == 1:
            decoder_input_ids = decoder_input_ids[None]
        seq_len = int(decoder_input_ids.shape[1])
        B = decoder_input_ids.shape[0]

        if past_key_values is None or seq_len > 1:
            logits = imp.prefill_full_sequence(decoder_input_ids)
            self.state = {"pos": [seq_len] * B, "batch": B}
            for b in range(B):
                imp.pos[b] = seq_len - 1
            return logits, self.state

        pos = self.state["pos"]
        for b in range(B):
            imp.pos[b] = pos[b]
        imp.x_in.copy_(decoder_input_ids)
        imp._run_body()
        self.state["pos"] = [p + 1 for p in pos]
        return imp.logits[:B].clone(), self.state

    @torch.inference_mode()
    def constrained_step(self, memory, token_in: int, grammar_state) -> int:
        imp = self._ensure(memory)
        mask = self.allowed_cache.tensor_for(grammar_state)
        pos = self.state["pos"]
        imp.pos[0] = pos if isinstance(pos, int) else pos[0]
        imp.x_in[0, 0] = token_in
        imp.allowed[0].copy_(mask)
        imp._run_body()
        if isinstance(pos, int):
            self.state["pos"] = pos + 1
        else:
            self.state["pos"][0] = (pos[0] if isinstance(pos, list) else pos) + 1
        return int(imp.token_out[0, 0].item())

    @torch.inference_mode()
    def constrained_step_batch(self, memory, tokens_in, grammar_states,
                               active=None):
        imp = self._ensure(memory)
        B = imp.B
        if memory.dim() == 2:
            memory = memory[None]
        if memory.shape[0] < B:  # pad to the exact bucket
            pad = memory[0:1].expand(B - memory.shape[0], -1, -1)
            memory = torch.cat([memory, pad], dim=0).contiguous()
        real = len(tokens_in)
        eos = self.model.tokenizer.eos_token
        tokens_in = list(tokens_in) + [eos] * (B - real)
        grammar_states = list(grammar_states) + [None] * (B - real)
        active = list(active) + [False] * (B - real) if active is not None \
            else None
        for b in range(B):
            pos_list = self.state["pos"]
            imp.pos[b] = pos_list[b] if b < len(pos_list) else pos_list[0]
            imp.x_in[b, 0] = tokens_in[b]
            gs = grammar_states[b] if (active is None or active[b]) else None
            if gs is None:
                # finished/padded slot: only eos is allowed, feed eos back
                mask = self.allowed_cache.tensor_for(
                    _EosOnlyState(self.model.tokenizer))
                imp.x_in[b, 0] = eos
            else:
                mask = self.allowed_cache.tensor_for(gs)
            imp.allowed[b].copy_(mask)
        imp._run_body()
        if len(self.state["pos"]) < B:
            self.state["pos"] = self.state["pos"] + \
                [self.state["pos"][0]] * (B - len(self.state["pos"]))
        self.state["pos"] = [p + 1 for p in self.state["pos"]]
        return [int(t) for t in imp.token_out[:, 0].tolist()]


class _EosOnlyState:
    """Minimal grammar-state stand-in whose allowed set is just {eos}."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def payload_count(self):
        return True  # can_end

    @property
    def in_shift(self):
        return False

    @property
    def shift_run(self):
        return 0

    @property
    def incomplete(self):
        return False

    @property
    def last_field_index(self):
        return -1

    def allowed(self, device):
        mask = torch.zeros(self.tokenizer.n_tokens, dtype=torch.bool,
                           device=device)
        mask[self.tokenizer.eos_token] = True
        return mask
