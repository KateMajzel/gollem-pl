#!/usr/bin/env python3
"""Tokenizuje korpus PL do formatu nanoGPT (.bin uint16) + meta.json.

Kluczowe: zapisuje total_utf8_bytes i bytes_per_token, zeby dalo sie ustawic
WSPOLNY BUDZET W BAJTACH miedzy przebiegami o roznych tokenizerach.

Wejscie: pliki .jsonl / .jsonl.gz, jeden dokument na linie, pole "text".

    python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer tokenizer.json --out data/pl_pltok
    python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer gpt2          --out data/pl_gpt2tok

Split train/val jest deterministyczny (hash doc_id), wiec oba przebiegi
widza DOKLADNIE ten sam podzial dokumentow.
"""
import argparse, glob, gzip, hashlib, io, json, os, sys
import numpy as np


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def load_tokenizer(spec):
    """spec: sciezka do tokenizer.json ALBO nazwa modelu HF (np. 'gpt2')."""
    from tokenizers import Tokenizer
    if os.path.exists(spec):
        tok = Tokenizer.from_file(spec)
        # separator dokumentow
        eot = tok.token_to_id("<|endoftext|>")
        if eot is None:
            raise SystemExit("Brak <|endoftext|> w tokenizerze - ustaw --eot-token recznie.")
    else:
        tok = Tokenizer.from_pretrained(spec)
        eot = tok.token_to_id("<|endoftext|>")
        if eot is None:
            eot = tok.get_vocab_size() - 1
    return tok, eot


def is_val(doc_id, val_frac):
    h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < val_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob, np. 'data/raw/*.jsonl.gz'")
    ap.add_argument("--tokenizer", required=True, help="sciezka do tokenizer.json lub nazwa HF")
    ap.add_argument("--out", required=True, help="katalog wyjsciowy")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=200, help="pomijaj krotkie smieci")
    ap.add_argument("--dedup", action="store_true", help="dedup dokumentow po SHA1 tekstu")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise SystemExit(f"Brak plikow dla globa: {args.input}")
    tok, eot = load_tokenizer(args.tokenizer)
    vocab = tok.get_vocab_size()
    if vocab >= 2 ** 16:
        raise SystemExit(f"vocab={vocab} nie miesci sie w uint16 - zmien dtype na uint32.")

    os.makedirs(args.out, exist_ok=True)
    handles = {s: open(os.path.join(args.out, f"{s}.bin"), "wb") for s in ("train", "val")}
    n_tok = {"train": 0, "val": 0}
    n_bytes = {"train": 0, "val": 0}
    n_docs = {"train": 0, "val": 0}
    seen = set()
    skipped = 0

    BUF = 8192
    buffers = {"train": [], "val": []}

    def flush(split):
        if buffers[split]:
            np.array(np.concatenate(buffers[split]), dtype=np.uint16).tofile(handles[split])
            buffers[split].clear()

    for fi, path in enumerate(files):
        with open_maybe_gz(path) as fh:
            for li, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)[args.text_field]
                except Exception:
                    skipped += 1
                    continue
                if not text or len(text) < args.min_chars:
                    skipped += 1
                    continue
                if args.dedup:
                    h = hashlib.sha1(" ".join(text.split()).encode()).digest()
                    if h in seen:
                        skipped += 1
                        continue
                    seen.add(h)

                split = "val" if is_val(f"{path}:{li}", args.val_frac) else "train"
                ids = tok.encode(text).ids
                ids.append(eot)  # separator dokumentow - BEZ TEGO model sklei niezalezne teksty
                buffers[split].append(np.array(ids, dtype=np.uint16))
                n_tok[split] += len(ids)
                n_bytes[split] += len(text.encode("utf-8"))
                n_docs[split] += 1
                if len(buffers[split]) >= BUF:
                    flush(split)
        print(f"[{fi+1}/{len(files)}] {path}  train={n_tok['train']:,} tok", flush=True)

    for s in handles:
        flush(s)
        handles[s].close()

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab,
        "eot_id": eot,
        "dtype": "uint16",
        "val_frac": args.val_frac,
        "docs": n_docs,
        "tokens": n_tok,
        "utf8_bytes": n_bytes,
        # <- TO jest liczba, ktora wyrownuje budzet miedzy tokenizerami
        "bytes_per_token": {s: (n_bytes[s] / n_tok[s] if n_tok[s] else 0) for s in n_tok},
        "skipped_docs": skipped,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    bpt = meta["bytes_per_token"]["train"]
    print(f"\n>>> bytes_per_token(train) = {bpt:.3f}")
    print(f">>> zeby zobaczyc B bajtow korpusu potrzebujesz B/{bpt:.3f} tokenow")


if __name__ == "__main__":
    main()
