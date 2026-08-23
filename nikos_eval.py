#!/usr/bin/env python3
"""Prywatny held-out: porownanie modeli na wlasnych, nieopublikowanych zdaniach.

Robi DWIE rzeczy:

  1. POMIAR (ilosciowy) — BPB odniesienia: jak bardzo model jest zaskoczony
     TWOIM prawdziwym zakonczeniem zdania. To pelnoprawna metryka, bo teksty
     nieopublikowane nie mogly trafic do zadnego korpusu treningowego —
     ani Twojego, ani GPT-2. Zero ryzyka kontaminacji.

  2. ILUSTRACJA (jakosciowa) — kontynuacje wygenerowane przez kazdy model,
     zestawione obok Twojego oryginalu do oceny wzrokowej.

Punkt 1 idzie do tabeli wynikow. Punkt 2 NIE jest metryka i nie wolno go tak
raportowac — sluzy do pokazania, czy roznica w liczbach jest w ogole widoczna.

    python nikos_eval.py --eval data/nikos_eval.jsonl \\
        --model R1=out/r1-tierB-pltok/ckpt.pt:tokenizer.json \\
        --model R2a=out/r2-tierB-gpt2tok/ckpt.pt:gpt2 \\
        --hf-model GPT2=gpt2 \\
        --out nikos_porownanie.md
"""
import argparse, json, math, os, sys
import torch


def get_tokenizer(spec):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(spec) if os.path.exists(spec) else Tokenizer.from_pretrained(spec)
    eot = tok.token_to_id("<|endoftext|>")
    return tok, (eot if eot is not None else tok.get_vocab_size() - 1)


def load_nanogpt(path, device):
    from model import GPT, GPTConfig
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(GPTConfig(**ck["model_args"]))
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    m.eval().to(device)
    return m, ck["model_args"]["block_size"]


def load_hf(name, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name).eval().to(device)
    return m, getattr(m.config, "n_positions", 1024)


@torch.no_grad()
def ref_nll(model, tok, eot, prompt, reference, device, is_hf):
    """NLL (naty) TWOJEGO zakonczenia + liczba jego bajtow."""
    cont = " " + reference if not prompt.endswith(" ") else reference
    p_ids = tok.encode(prompt).ids
    c_ids = tok.encode(cont).ids
    x = torch.tensor([[eot] + p_ids + c_ids], dtype=torch.long, device=device)
    logits = model(x).logits[0, :-1] if is_hf else model(x[:, :-1], x[:, 1:])[0][0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = x[0, 1:]
    n = len(c_ids)
    nll = -lp[-n:].gather(1, tgt[-n:].unsqueeze(1)).sum().item()
    return nll, len(cont.encode("utf-8")), n


@torch.no_grad()
def generate(model, tok, eot, prompt, device, is_hf, max_new=40, temp=0.8, top_k=50, seed=0):
    torch.manual_seed(seed)
    ids = [eot] + tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(x).logits[:, -1, :] if is_hf else model(x)[0][:, -1, :]
        logits = logits.float() / temp
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if int(nxt) == eot:
            break
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0, len(ids):].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/nikos_eval.jsonl")
    ap.add_argument("--model", action="append", default=[],
                    help="NAZWA=sciezka/ckpt.pt:tokenizer  (nanoGPT)")
    ap.add_argument("--hf-model", action="append", default=[],
                    help="NAZWA=id_lub_sciezka  (HuggingFace)")
    ap.add_argument("--out", default="nikos_porownanie.md")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    print(f"prywatny held-out: {len(items)} zdan")
    if len(items) < 30:
        print("UWAGA: przy <30 zdaniach BPB odniesienia ma szeroki przedzial ufnosci.")
        print("       Traktuj jako sygnal kierunkowy, nie rozstrzygniecie.\n")

    specs = []
    for s in a.model:
        name, rest = s.split("=", 1)
        ckpt, tokspec = rest.rsplit(":", 1)
        specs.append((name, ckpt, tokspec, False))
    for s in a.hf_model:
        name, mid = s.split("=", 1)
        specs.append((name, mid, mid, True))
    if not specs:
        raise SystemExit("Podaj co najmniej jeden --model lub --hf-model")

    results, gens = {}, {}
    for name, path, tokspec, is_hf in specs:
        print(f"== {name}")
        model, block = (load_hf(path, a.device) if is_hf else load_nanogpt(path, a.device))
        tok, eot = get_tokenizer(tokspec)
        tot_nll = tot_b = tot_t = 0
        outs = []
        for it in items:
            nll, nb, nt = ref_nll(model, tok, eot, it["prompt"], it["reference"], a.device, is_hf)
            tot_nll += nll; tot_b += nb; tot_t += nt
            outs.append(generate(model, tok, eot, it["prompt"], a.device, is_hf,
                                 a.max_new, a.temp, seed=a.seed))
        bpb = tot_nll / (math.log(2) * tot_b)
        results[name] = {"bpb_ref": bpb, "bytes": tot_b, "tokens": tot_t,
                         "bytes_per_token": tot_b / tot_t}
        gens[name] = outs
        print(f"   BPB odniesienia = {bpb:.4f}  ({tot_b} B, {tot_t} tok.)")
        del model
        torch.cuda.empty_cache()

    # --- raport ---
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# Prywatny held-out — porównanie modeli\n\n")
        f.write(f"Zdań: {len(items)} · nieopublikowane, więc żaden model ich nie widział.\n\n")
        f.write("## 1. Pomiar: BPB odniesienia\n\n")
        f.write("Jak bardzo model jest zaskoczony *prawdziwym* zakończeniem zdania. Niżej = lepiej.\n\n")
        f.write("| model | BPB | bajty/token |\n|---|---|---|\n")
        for n, r in sorted(results.items(), key=lambda kv: kv[1]["bpb_ref"]):
            f.write(f"| {n} | {r['bpb_ref']:.4f} | {r['bytes_per_token']:.2f} |\n")
        f.write(f"\nŁącznie {list(results.values())[0]['bytes']} bajtów odniesienia "
                f"— identycznych dla wszystkich modeli.\n\n")
        f.write("## 2. Ilustracja: wygenerowane kontynuacje\n\n")
        f.write("**To nie jest metryka.** Służy sprawdzeniu, czy różnice z punktu 1 są widoczne.\n\n")
        for i, it in enumerate(items):
            f.write(f"### {i+1}. `{it['prompt']}`\n\n")
            f.write(f"- **oryginał** — {it['reference']}\n")
            for n in results:
                f.write(f"- **{n}** — {gens[n][i].strip()}\n")
            f.write("\n")
    print(f"\n-> {a.out}")
    print("\nBPB odniesienia (niżej = lepiej):")
    for n, r in sorted(results.items(), key=lambda kv: kv[1]["bpb_ref"]):
        print(f"  {n:<8} {r['bpb_ref']:.4f}")


if __name__ == "__main__":
    main()
