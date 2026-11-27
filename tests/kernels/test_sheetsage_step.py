"""Test of ASR decode

Run : python -m pytest \
        tests/kernels/test_sheetsage_step.py -q -s
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veloxvoice.kernels.ops.music.cross_attn import cross_attn_decode  # noqa: E402

from veloxvoice.kernels.ops.music.cuda_ops import (  # noqa: E402
    ss_gemv, 
    ss_gemv_ln, 
    ss_kvstore, 
    ss_ln_res, 
    ss_lmhead, 
    ss_self_attn,
    ss_self_attn_split)

from veloxvoice.models.sheetsage.v2.step_pipeline import (  # noqa: E402
    LAYER_ELEMS, 
    MultiKernelStepDecoder, 
    pack_decoder_weights)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="requires CUDA")

D, H, MODEL, QKV, FFN = 64, 8, 512, 1536, 2048
SONG = "test_data/music/bohemian_rhapsody_queen.mp3"


#   MODEL_DIR        SheetSage2 adapter dir
#   BASE_MODEL_DIR   MERT-v2-FullSong base dir
#   SHEETSAGE_SONG   test audio
MODEL_DIR = os.environ.get("MODEL_DIR", "/raid/yiakwy/SheetSage2")
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "/raid/yiakwy/MERT-v2-FullSong")
SONG = os.environ.get("SHEETSAGE_SONG",
                      "test_data/music/bohemian_rhapsody_queen.mp3")


@pytest.fixture(scope="module")
def ctx():
    from veloxvoice.models.sheetsage.v2 import api as sheetsage_api
    from veloxvoice.stream.grammar_ar_decode import (
        FULL_TASK_PROMPTS, 
        build_grammar_tables)

    m = sheetsage_api.load_model(MODEL_DIR,
                                 BASE_MODEL_DIR, "bf16", "cuda")
                                 
    audio = torch.zeros(1, 300 * 24000, device="cuda")
    prompts = m.tokenizer.normalize_prompts(FULL_TASK_PROMPTS)

    prefix = m.tokenizer.prompt_prefix(prompts)

    from veloxvoice.models.sheetsage.v2.audio_loader import load_audio, SAMPLE_RATE

    wav = load_audio(str(ROOT / SONG), sampling_rate=SAMPLE_RATE)
    wav = wav[0] if isinstance(wav, tuple) else wav

    seg = wav[:int(300 * SAMPLE_RATE)].reshape(1, -1).contiguous().cuda()

    mem = m.encode(seg)

    g = m._graphed
    with torch.inference_mode():
        imp = g._ensure(mem.contiguous())
        imp.prefill_full_sequence(
            torch.tensor([prefix], dtype=torch.long, device="cuda"))
        imp.refresh_cross(mem.contiguous())
    
    at, sm, _ = build_grammar_tables(m.tokenizer, prompts, "cuda")
    w_blob = pack_decoder_weights(m.decoder)
    L, S, T = 6, imp.S, imp.T

    deco = MultiKernelStepDecoder(
        m, 
        imp.k_cross.view(L, H, T, D), imp.v_cross.view(L, H, T, D),
        imp.k_self.view(L, H, S, D), imp.v_self.view(L, H, S, D),
        w_blob, 
        at, sm, 
        m.config.vocab_size, m.tokenizer.eos_token)
    return dict(m=m, imp=imp, deco=deco, at=at, sm=sm, prefix=prefix, mem=mem)


def _rep(tag, a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    d = (a - b).abs().max().item()
    print(f"  {tag:24s} max|d|={d:.5f}  |ref|={b.abs().mean().item():.4f} "
          f"|got|={a.abs().mean().item():.4f}")
    return d


def test_stage_parity(ctx):
    m, imp, deco = ctx["m"], ctx["imp"], ctx["deco"]
    tok, pos, l = ctx["prefix"][-1], 8, 0
    v = deco._lviews[l]
    lns = v["lns"]

    with torch.inference_mode():
        deco.pos_dev.fill_(pos)
        deco.x_in.fill_(tok)
        deco._embed()
        torch.cuda.synchronize()

        e = F.embedding(torch.tensor([tok], device="cuda"), deco.embed_w)
        pe = F.embedding(torch.tensor([pos + 2], device="cuda"), deco.pos_table)
        h = F.layer_norm((e[0] + pe[0]).float(), (MODEL,),
                         deco.ln_emb_w.float(), deco.ln_emb_b.float(), 1e-5)
        d_embed = _rep("embed", deco.h, h)
        assert d_embed < 0.02

        # attn gemv
        ss_gemv(v["qkv_w"], v["qkv_b"], deco.h, deco.qkv_buf, QKV, MODEL, 64)
        torch.cuda.synchronize()
        qkv = F.linear(h, v["qkv_w"].float(), v["qkv_b"].float())
        assert _rep("qkv_gemv", deco.qkv_buf, qkv) < 0.05

        # kvstore + self_attn
        ss_kvstore(deco.qkv_buf, deco.k_self[l], deco.v_self[l], deco.S,
                   deco.pos_dev)
        ss_self_attn(deco.qkv_buf, deco.k_self[l], deco.v_self[l],
                     deco.s_attn, deco.S, deco.pos_dev, D ** -0.5, H)
        torch.cuda.synchronize()

        q, kk, vv = qkv.split(MODEL)
        Ks, Vs = deco.k_self[l].float(), deco.v_self[l].float()

        Ks[:, pos, :] = kk.view(H, D)
        Vs[:, pos, :] = vv.view(H, D)
        s_ref = F.scaled_dot_product_attention(
            q.float().view(1, H, 1, D), Ks[:, :pos + 1].unsqueeze(0),
            Vs[:, :pos + 1].unsqueeze(0), scale=D ** -0.5)[0, :, 0, :]
        assert _rep("self_attn", deco.s_attn.view(H, D), s_ref) < 0.05

        # out gemv + residual + LN
        ss_gemv(v["out_w"], v["out_b"], deco.s_attn, deco.tmp, MODEL, MODEL, 32)
        ss_ln_res(deco.h, deco.tmp, lns[0:MODEL], lns[MODEL:2 * MODEL])
        torch.cuda.synchronize()

        h = F.layer_norm(h + F.linear(s_ref.reshape(-1), v["out_w"].float(),
                                      v["out_b"].float()), (MODEL,),
                         lns[0:MODEL].float(), lns[MODEL:2 * MODEL].float(), 1e-5)
        assert _rep("self_out+ln", deco.h, h) < 0.05

        # xq gemv
        ss_gemv(v["xq_w"], v["xq_b"], deco.h, deco.q_layer, MODEL, MODEL, 32)
        torch.cuda.synchronize()
        qx = F.linear(h, v["xq_w"].float(), v["xq_b"].float())
        assert _rep("xq_gemv", deco.q_layer, qx) < 0.05

        # cross attn
        deco.a_layer.copy_(cross_attn_decode(
            deco.q_layer.view(H, D), deco.k_cross, deco.v_cross, l).reshape(-1))
        torch.cuda.synchronize()

        Kc = deco.k_cross[l].float()
        Vc = deco.v_cross[l].float()

        a_ref = F.scaled_dot_product_attention(
            qx.view(1, H, 1, D), Kc.unsqueeze(0), Vc.unsqueeze(0),
            scale=D ** -0.5)[0, :, 0, :]
        assert _rep("cross_attn", deco.a_layer.view(H, D), a_ref) < 0.05

        # cross out + residual + LN (encoder_attn.out_proj == xo_w)
        ss_gemv(v["xo_w"], v["xo_b"], deco.a_layer, deco.tmp, MODEL, MODEL, 32)
        ss_ln_res(deco.h, deco.tmp, lns[2 * MODEL:3 * MODEL],
                  lns[3 * MODEL:4 * MODEL])
        torch.cuda.synchronize()

        h = F.layer_norm(h + F.linear(a_ref.reshape(-1), v["xo_w"].float(),
                                      v["xo_b"].float()), (MODEL,),
                         lns[2 * MODEL:3 * MODEL].float(),
                         lns[3 * MODEL:4 * MODEL].float(), 1e-5)
        assert _rep("cross_out+ln", deco.h, h) < 0.05

        # ffn
        ss_gemv(v["f1_w"], v["f1_b"], deco.h, deco.ffn_buf, FFN, MODEL, 96)
        F.gelu(deco.ffn_buf, out=deco.ffn_buf)

        ss_gemv(v["f2_w"], v["f2_b"], deco.ffn_buf, deco.tmp, MODEL, FFN, 32)
        ss_ln_res(deco.h, deco.tmp, lns[4 * MODEL:5 * MODEL],
                  lns[5 * MODEL:6 * MODEL])
        torch.cuda.synchronize()

        ff = F.linear(F.gelu(F.linear(h, v["f1_w"].float(), v["f1_b"].float())),
                      v["f2_w"].float(), v["f2_b"].float())
        h = F.layer_norm(h + ff, (MODEL,), lns[4 * MODEL:5 * MODEL].float(),
                         lns[5 * MODEL:6 * MODEL].float(), 1e-5)
        assert _rep("ffn+ln", deco.h, h) < 0.05

        # lm_head
        deco.allowed_cur.copy_(deco.allowed_table[0])
        ss_lmhead(deco.lm_head, deco.h, deco.allowed_cur, deco.lm_part_b,
                  deco.lm_part_i, deco.V, MODEL, 64)
        torch.cuda.synchronize()

        b = int(deco.lm_part_b.argmax())
        got = int(deco.lm_part_i[b])

        lg = F.linear(h, deco.lm_head.float()).masked_fill(
            ~deco.allowed_cur.bool(), float("-inf"))
        ref = int(lg.argmax())

        print(f"  lm_head argmax: got={got} ref={ref}")
        assert got == ref

def test_split_self_attn_parity(ctx):
    """Split-K self-attn"""
    deco = ctx["deco"]
    l, pos = 0, 8
    split = 16

    pm = torch.zeros(H, split, dtype=torch.float32, device="cuda")
    pl = torch.zeros(H, split, dtype=torch.float32, device="cuda")
    pacc = torch.zeros(H, D, split, dtype=torch.float32, device="cuda")
    out = torch.zeros(H * D, dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        deco.pos_dev.fill_(pos)
        deco.x_in.fill_(ctx["prefix"][-1])
        deco._embed()
        # build a realistic qkv + cache
        deco._layer(0)   # leaves caches / qkv populated
        deco.pos_dev.fill_(pos)

        ss_self_attn_split(deco.qkv_buf, deco.k_self[l], deco.v_self[l], out,
                           pm, pl, pacc, deco.S, deco.pos_dev, split, H,
                           D ** -0.5)
        torch.cuda.synchronize()

        q = deco.qkv_buf[:MODEL].float().view(H, D)
        Ks, Vs = deco.k_self[l].float(), deco.v_self[l].float()
        s_ref = F.scaled_dot_product_attention(
            q.view(1, H, 1, D), Ks[:, :pos + 1].unsqueeze(0),
            Vs[:, :pos + 1].unsqueeze(0), scale=D ** -0.5)[0, :, 0, :]
        assert _rep("split_self_attn", out.view(H, D), s_ref) < 0.05


def test_full_step_parity_vs_model(ctx):
    """End-to-end 6-layer ASR decode step"""
    m, imp, deco = ctx["m"], ctx["imp"], ctx["deco"]
    pos, H, D, L = 8, 8, 64, 6

    with torch.inference_mode():
        deco.pos_dev.fill_(pos)
        deco.x_in.fill_(ctx["prefix"][-1])
        deco._embed()
        torch.cuda.synchronize()
        h0 = deco.h.float().clone()

        Ks = imp.k_self.view(L, H, imp.S, D).float().clone()
        Vs = imp.v_self.view(L, H, imp.S, D).float().clone()
        Kc = imp.k_cross.view(L, H, imp.T, D).float()
        Vc = imp.v_cross.view(L, H, imp.T, D).float()

        def lin(x, mod):
            return F.linear(x, mod.weight.float(), mod.bias.float())

        ref_cross = []
        h = h0
        for l, layer in enumerate(m.decoder.layers):
            q = lin(h, layer.self_attn.q_proj)
            k = lin(h, layer.self_attn.k_proj)
            v = lin(h, layer.self_attn.v_proj)

            Ks[l][:, pos, :] = k.view(H, D)
            Vs[l][:, pos, :] = v.view(H, D)

            a = F.scaled_dot_product_attention(
                q.view(1, H, 1, D), Ks[l][:, :pos + 1].unsqueeze(0),
                Vs[l][:, :pos + 1].unsqueeze(0), scale=D ** -0.5)[0, :, 0, :]
            
            h = h + lin(a.reshape(-1), layer.self_attn.out_proj)
            h = F.layer_norm(h, (MODEL,),
                             layer.self_attn_layer_norm.weight.float(),
                             layer.self_attn_layer_norm.bias.float(), 1e-5)

            qx = lin(h, layer.encoder_attn.q_proj)
            a = F.scaled_dot_product_attention(
                qx.view(1, H, 1, D), Kc[l].unsqueeze(0), Vc[l].unsqueeze(0),
                scale=D ** -0.5)[0, :, 0, :]
            h = h + lin(a.reshape(-1), layer.encoder_attn.out_proj)
            h = F.layer_norm(h, (MODEL,),
                             layer.encoder_attn_layer_norm.weight.float(),
                             layer.encoder_attn_layer_norm.bias.float(), 1e-5)
            
            ref_cross.append(h.clone())
            ff = lin(F.gelu(lin(h, layer.fc1)), layer.fc2)
            h = F.layer_norm(h + ff, (MODEL,),
                             layer.final_layer_norm.weight.float(),
                             layer.final_layer_norm.bias.float(), 1e-5)
    
        for l in range(L):
            deco._layer(l)
            torch.cuda.synchronize()

            d = _rep(f"layer{l} h", deco.h, ref_cross[l])
            assert d < 0.15, f"layer {l} full-step h parity broke: {d}"
        
        deco._final_ln()
        torch.cuda.synchronize()
        assert _rep("final h", deco.h, h) < 0.15

        # lm_head argmax
        deco.allowed_cur.copy_(deco.allowed_table[0])
        ss_lmhead(deco.lm_head, deco.h, deco.allowed_cur, deco.lm_part_b,
                  deco.lm_part_i, deco.V, MODEL, 64)
        torch.cuda.synchronize()
        
        got = int(deco.lm_part_i[int(deco.lm_part_b.argmax())])
        lg = F.linear(h, deco.lm_head.float()).masked_fill(
            ~deco.allowed_cur.bool(), float("-inf"))
        ref = int(lg.argmax())
        print(f"  full-step lm_head argmax: got={got} ref={ref}")
        assert got == ref, "full-step argmax mismatch — weight assembly bug?"
