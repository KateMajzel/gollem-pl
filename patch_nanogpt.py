#!/usr/bin/env python3
"""Naklada patche na train.py i sample.py z nanoGPT pod wlasny tokenizer PL.

    python patch_nanogpt.py            # DRY-RUN: pokazuje, co zmieni, nic nie zapisuje
    python patch_nanogpt.py --apply    # zapisuje zmiany (robi kopie .bak)

Cztery patche:
  1. train.py  — meta.json zamiast meta.pkl (+ odczyt bytes_per_token)
  2. train.py  — logowanie przebytych GB korpusu obok loss
  3. train.py  — ckpt_last.pt co 250 iteracji (crash-safe przy dlugim runie)
  4. sample.py — wlasny tokenizer zamiast tiktoken/GPT-2

Skrypt jest idempotentny: uruchomiony drugi raz wykryje, ze patch juz jest.
"""
import argparse, os, shutil, sys


def find_span(lines, start_pred, end_pred, name):
    """Zwraca (i, j) — indeksy pierwszej i ostatniej linii bloku (wlacznie)."""
    i = next((k for k, l in enumerate(lines) if start_pred(l)), None)
    if i is None:
        return None
    j = next((k for k in range(i, len(lines)) if end_pred(lines[k])), None)
    if j is None:
        raise SystemExit(f"[{name}] znaleziono poczatek bloku, ale nie koniec — przerywam.")
    return i, j


P1_NEW = '''# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.json')
meta_vocab_size = None
bytes_per_token = 0.0
if os.path.exists(meta_path):
    import json as _json
    with open(meta_path) as f:
        meta = _json.load(f)
    meta_vocab_size = meta['vocab_size']
    bytes_per_token = meta['bytes_per_token']['train']
    print(f"found vocab_size = {meta_vocab_size}, bytes_per_token = {bytes_per_token:.3f} (inside {meta_path})")
'''

P2_NEW = '''        gb_seen = iter_num * tokens_per_iter * bytes_per_token / 1e9
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, {gb_seen:.2f} GB")
'''

P3_NEW = '''    # crash-safe: zapis co 250 iteracji NIEZALEZNIE od val loss
    if iter_num % 250 == 0 and iter_num > 0 and master_process:
        torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': model_args, 'iter_num': iter_num,
                    'best_val_loss': best_val_loss, 'config': config},
                   os.path.join(out_dir, 'ckpt_last.pt'))

'''

P4_NEW = '''# wlasny tokenizer PL (zamiast tiktoken/GPT-2)
from tokenizers import Tokenizer as _Tok
_TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
_tok = _Tok.from_file(_TOKENIZER_PATH)
print(f"Using tokenizer: {_TOKENIZER_PATH} (vocab {_tok.get_vocab_size()})")
encode = lambda s: _tok.encode(s).ids
decode = lambda l: _tok.decode(l)
'''


def patch_train(path, apply, log):
    lines = open(path, encoding='utf-8').read().splitlines(keepends=True)
    changed = 0

    # --- P1: meta.pkl -> meta.json ---
    if any("meta.json" in l for l in lines):
        log("  [1] meta.json — JUZ NALOZONY, pomijam")
    else:
        span = find_span(lines,
                         lambda l: l.startswith("meta_path = os.path.join(data_dir, 'meta.pkl')"),
                         lambda l: l.lstrip().startswith('print(f"found vocab_size'),
                         "P1")
        if not span:
            raise SystemExit("[1] nie znalazlem bloku meta.pkl w train.py")
        i, j = span
        # zjadamy tez poprzedzajacy komentarz, jesli jest
        if i > 0 and lines[i-1].startswith("# attempt to derive vocab_size"):
            i -= 1
        log(f"  [1] meta.pkl -> meta.json  (linie {i+1}-{j+1})")
        lines[i:j+1] = [P1_NEW]
        changed += 1

    # --- P2: logowanie GB ---
    if any("gb_seen" in l for l in lines):
        log("  [2] logowanie GB — JUZ NALOZONY, pomijam")
    else:
        k = next((k for k, l in enumerate(lines)
                  if l.lstrip().startswith('print(f"iter {iter_num}: loss')), None)
        if k is None:
            raise SystemExit("[2] nie znalazlem linii z logowaniem loss w train.py")
        log(f"  [2] logowanie przebytych GB  (linia {k+1})")
        lines[k:k+1] = [P2_NEW]
        changed += 1

    # --- P3: ckpt_last.pt ---
    if any("ckpt_last.pt" in l for l in lines):
        log("  [3] ckpt_last.pt — JUZ NALOZONY, pomijam")
    else:
        k = next((k for k, l in enumerate(lines)
                  if l.startswith("    if iter_num % log_interval == 0 and master_process:")), None)
        if k is None:
            raise SystemExit("[3] nie znalazlem petli logowania w train.py")
        log(f"  [3] ckpt_last.pt co 250 iteracji  (wstawka przed linia {k+1})")
        lines[k:k] = [P3_NEW]
        changed += 1

    if apply and changed:
        shutil.copy(path, path + ".bak")
        open(path, "w", encoding='utf-8').write("".join(lines))
    return changed


def patch_sample(path, apply, log):
    lines = open(path, encoding='utf-8').read().splitlines(keepends=True)
    if any("_TOKENIZER_PATH" in l for l in lines):
        log("  [4] tokenizer w sample.py — JUZ NALOZONY, pomijam")
        return 0

    span = find_span(lines,
                     lambda l: l.startswith("load_meta = False"),
                     lambda l: l.lstrip().startswith("decode = lambda l: enc.decode(l)"),
                     "P4")
    if not span:
        raise SystemExit("[4] nie znalazlem bloku tokenizera w sample.py")
    i, j = span
    if i > 0 and lines[i-1].startswith("# look for the meta pickle"):
        i -= 1
    log(f"  [4] wlasny tokenizer zamiast tiktoken  (linie {i+1}-{j+1})")
    lines[i:j+1] = [P4_NEW]

    # tiktoken nie jest juz potrzebny
    lines = [l for l in lines if l.strip() != "import tiktoken"]

    if apply:
        shutil.copy(path, path + ".bak")
        open(path, "w", encoding='utf-8').write("".join(lines))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="zapisz zmiany (bez tego tylko podglad)")
    ap.add_argument("--dir", default=".", help="katalog z train.py i sample.py")
    a = ap.parse_args()

    train_p = os.path.join(a.dir, "train.py")
    sample_p = os.path.join(a.dir, "sample.py")
    for p in (train_p, sample_p):
        if not os.path.exists(p):
            raise SystemExit(f"Nie znaleziono {p} — jestes w katalogu nanoGPT?")

    log = print
    print("train.py:")
    n1 = patch_train(train_p, a.apply, log)
    print("sample.py:")
    n2 = patch_sample(sample_p, a.apply, log)

    if not a.apply:
        print(f"\nDRY-RUN — nic nie zapisano. Zmian do naniesienia: {n1+n2}")
        print("Zeby zapisac:  python patch_nanogpt.py --apply")
    else:
        print(f"\nZapisano. Kopie zapasowe: train.py.bak, sample.py.bak")
        print("Sprawdz:  git diff")
        print("Cofniecie: git checkout train.py sample.py")


if __name__ == "__main__":
    main()
