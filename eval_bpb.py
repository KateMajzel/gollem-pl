#!/usr/bin/env python3
"""Bits-per-byte: uczciwe porownanie modeli o ROZNYCH tokenizerach.

Perplexity per token jest NIEPOROWNYWALNE miedzy tokenizerami - model z gestszym
tokenizerem przewiduje mniej, ale trudniejszych jednostek. BPB normalizuje po
bajtach UTF-8 tekstu, ktore sa wspolne dla wszystkich modeli.

    BPB = suma_NLL_w_natach / (ln(2) * liczba_bajtow_UTF-8)

Uzycie:
    # model nanoGPT (checkpoint z train.py) + wlasny tokenizer
    python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r1/ckpt.pt --tokenizer tokenizer.json --label R1-pltok

    # model nanoGPT trenowany na tokenizerze GPT-2
    python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2/ckpt.pt --tokenizer gpt2 --label R2-gpt2tok

    # pretrained GPT-2 zero-shot
    python eval_bpb.py --eval data/heldout.jsonl --hf gpt2 --label R3-gpt2-zeroshot

    # zbiorcza tabela z wielu przebiegow
    python eval_bpb.py --report results/*.json
"""
import argparse, glob, json, math, os, sys
import torch


def load_eval(path, field="text", limit=0):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get(field)
            if t:
                docs.append(t)
            if limit and len(docs) >= limit:
                break
    return docs


def load_nanogpt(ckpt_path, device):
    """Wczytuje checkpoint nanoGPT. Wymaga model.py z repo nanoGPT na sciezce."""
    from model import GPT, GPTConfig
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    conf = GPTConfig(**ck["model_args"])
    model = GPT(conf)
    sd = ck["model"]
    # torch.compile dokleja prefiks
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, conf.block_size


def load_hf(name, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(device)
    return model, model.config.n_positions if hasattr(model.config, "n_positions") else model.config.max_position_embeddings


def get_tokenizer(spec):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(spec) if os.path.exists(spec) else Tokenizer.from_pretrained(spec)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = tok.get_vocab_size() - 1
    return tok, eot


def clip_utf8(text, max_bytes):
    """Przycina tekst do max_bytes, nie rozcinajac znaku wielobajtowego."""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    return b[:max_bytes].decode("utf-8", errors="ignore")


@torch.no_grad()
def score(model, tok, eot, docs, block_size, device, is_hf, max_bytes):
    """Zwraca (suma_NLL_natow, suma_bajtow, suma_tokenow).

    KLUCZOWE DLA UCZCIWOSCI: przycinamy po BAJTACH, nie po tokenach.
    Przyciecie do block_size tokenow dawaloby kazdemu modelowi INNA ilosc
    tekstu (tokenizer PL upakowuje ~2x wiecej bajtow w token), wiec BPB
    liczyloby sie na roznych fragmentach i porownanie byloby bez sensu.
    max_bytes musi byc na tyle male, by zmiescic sie w kontekscie modelu
    o NAJRZADSZYM tokenizerze (GPT-2 na PL: ~2.07 B/token).

    Dla kazdego dokumentu:
      - przycinamy tekst do max_bytes,
      - tokenizujemy; jesli mimo to za dlugi, tniemy tokeny i przeliczamy
        bajty przez decode (byte-level BPE jest bezstratne),
      - poprzedzamy <|endoftext|> i liczymy NLL na wszystkich tokenach tresci.
    """
    total_nll, total_bytes, total_tokens = 0.0, 0, 0
    n_clipped = 0
    per_doc = []          # (indeks, nll, bajty, tokeny) — potrzebne do bootstrapu
    for i, text in enumerate(docs):
        text = clip_utf8(text, max_bytes)
        ids = tok.encode(text).ids
        if len(ids) > block_size - 1:          # awaryjnie, gdy tekst gesty
            ids = ids[: block_size - 1]
            text = tok.decode(ids)
            n_clipped += 1
        if len(ids) < 2:
            continue
        total_bytes += len(text.encode("utf-8"))
        x = torch.tensor([[eot] + ids], dtype=torch.long, device=device)
        if is_hf:
            logits = model(x).logits[0, :-1]
        else:
            # nanoGPT bez targets zwraca logity TYLKO dla ostatniej pozycji;
            # z targets liczy pelne logity dla calej sekwencji
            logits = model(x[:, :-1], x[:, 1:])[0][0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = x[0, 1:]
        d_nll = -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        total_nll += d_nll
        total_tokens += tgt.numel()
        per_doc.append((i, d_nll, len(text.encode("utf-8")), int(tgt.numel())))
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(docs)}", flush=True)
    if n_clipped:
        print(f"  (uwaga: {n_clipped} dok. przycietych dodatkowo po tokenach)", flush=True)
    return total_nll, total_bytes, total_tokens, per_doc


def report(paths):
    rows = []
    for p in paths:
        for q in glob.glob(p):
            rows.append(json.load(open(q)))
    if not rows:
        return
    rows.sort(key=lambda r: r["bpb"])
    w = max(len(r["label"]) for r in rows)
    print(f"\n{'run':<{w}}  {'BPB':>7}  {'PPL/tok':>9}  {'bajty/tok':>9}  {'tokeny':>10}")
    print("-" * (w + 42))
    for r in rows:
        print(f"{r['label']:<{w}}  {r['bpb']:>7.4f}  {r['ppl_token']:>9.2f}  "
              f"{r['bytes_per_token']:>9.2f}  {r['tokens']:>10,}")
    print("\nBPB: nizej = lepiej, POROWNYWALNE miedzy tokenizerami.")
    print("PPL/tok: NIEporownywalne miedzy tokenizerami - tylko informacyjnie.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="*", help="zloz tabele z zapisanych wynikow .json")
    ap.add_argument("--eval", help="plik .jsonl z held-outem")
    ap.add_argument("--ckpt", help="checkpoint nanoGPT")
    ap.add_argument("--hf", help="model HF, np. gpt2 albo flax-community/papuGaPT2")
    ap.add_argument("--tokenizer", help="tokenizer.json lub nazwa HF (dla --ckpt)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=1800,
                    help="wspolny budzet bajtow na dokument; musi zmiescic sie w kontekscie "
                         "modelu o najrzadszym tokenizerze (GPT-2 na PL ~2.07 B/tok -> 1023 tok ~ 2100 B)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.report:
        return report(a.report)
    if not a.eval or not (a.ckpt or a.hf):
        ap.error("podaj --eval oraz --ckpt lub --hf (albo --report)")

    docs = load_eval(a.eval, limit=a.limit)
    print(f"held-out: {len(docs)} dokumentow, budzet {a.max_bytes} B/dok")

    if a.ckpt:
        if not a.tokenizer:
            ap.error("--ckpt wymaga --tokenizer")
        model, block = load_nanogpt(a.ckpt, a.device)
        tok, eot = get_tokenizer(a.tokenizer)
        is_hf = False
    else:
        model, block = load_hf(a.hf, a.device)
        tok, eot = get_tokenizer(a.tokenizer or a.hf)
        is_hf = True

    nll, nbytes, ntok, per_doc = score(model, tok, eot, docs, block, a.device, is_hf, a.max_bytes)
    bpb = nll / (math.log(2) * nbytes)
    res = {
        "label": a.label,
        "model": a.ckpt or a.hf,
        "tokenizer": a.tokenizer or a.hf,
        "eval": a.eval,
        "docs": len(docs),
        "max_bytes_per_doc": a.max_bytes,
        "bpb": bpb,
        "ppl_token": math.exp(nll / ntok),
        "bytes_per_token": nbytes / ntok,
        "tokens": ntok,
        "bytes": nbytes,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.label}.json")
    json.dump(res, open(out, "w"), indent=2)
    pd_out = os.path.join(a.out_dir, f"{a.label}_perdoc.json")
    json.dump({"label": a.label, "max_bytes": a.max_bytes, "per_doc": per_doc},
              open(pd_out, "w"))
    print(json.dumps(res, indent=2))
    print(f"-> {out}")
    print(f"-> {pd_out}  ({len(per_doc)} dok., do bootstrapu)")


if __name__ == "__main__":
    main()
