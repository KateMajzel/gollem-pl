#!/usr/bin/env python3
"""Benchmarki klasyfikacyjne PL metoda answer-log-likelihood (bez generowania).

Model nie generuje odpowiedzi — dostaje kilka wariantow zakonczenia i wybiera
najbardziej prawdopodobny. Dziala nawet dla bardzo malych modeli, ktore nie
umialyby wygenerowac poprawnej etykiety.

    python bench_pl.py --task polemo2 --ckpt out/r1-tierB-pltok/ckpt.pt --tokenizer tokenizer.json --label R1
    python bench_pl.py --task 8tags   --ckpt out/r2-tierB-gpt2tok/ckpt.pt --tokenizer gpt2 --label R2a
    python bench_pl.py --task polemo2 --hf gpt2 --label R3
    python bench_pl.py --report 'bench/*.json'

UCZCIWOSC MIEDZY TOKENIZERAMI:
  - tekst przycinany po BAJTACH (nie tokenach), tak jak w eval_bpb.py;
  - etykiety to te same ciagi znakow dla wszystkich modeli;
  - punktacja normalizowana po bajtach etykiety, bo etykiety o roznej dlugosci
    tokenowej inaczej obciazalyby sume logprobow. Raportujemy obie wersje.

OGRANICZENIE: model 42M na zadaniach MCQ bywa blisko poziomu losowego.
Interesuje nas ROZNICA miedzy modelami, nie poziom bezwzgledny.
"""
import argparse, glob, json, math, os, random, sys
import torch

# ---------------------------------------------------------------- zadania

# Nazwy klas czytamy z ClassLabel zbioru (kolejnosc jest tam autorytatywna),
# a nie z zakodowanej listy — zgadywanie kolejnosci to prosta droga do wyniku
# na poziomie losowym przy poprawnie dzialajacym modelu.
EN2PL = {
    "film": "film", "history": "historia", "food": "jedzenie", "medicine": "medycyna",
    "motorization": "motoryzacja", "work": "praca", "sport": "sport",
    "technology": "technologia",
}

TASKS = {
    "polemo2": {
        # sentyment recenzji; 4 klasy
        "hf_ids": ["allegro/klej-polemo2-in", "clarin-pl/polemo2-official"],
        "split": "test",
        "text_field": ["sentence", "text"],
        "label_field": ["target", "label"],
        "labels": {                       # mapowanie etykiety zbioru -> slowo PL
            "__label__meta_plus_m": "pozytywny",
            "__label__meta_minus_m": "negatywny",
            "__label__meta_zero": "neutralny",
            "__label__meta_amb": "niejednoznaczny",
            "0": "negatywny", "1": "neutralny", "2": "pozytywny", "3": "niejednoznaczny",
        },
        "template": "Recenzja: {text}\nWydźwięk tej recenzji jest",
        "choices": ["pozytywny", "negatywny", "neutralny", "niejednoznaczny"],
    },
    "8tags": {
        # klasyfikacja tematu naglowka; 8 klas
        "hf_ids": ["sdadas/8tags"],
        "split": "test",
        "text_field": ["sentence", "text"],
        "label_field": ["label", "target"],
        "labels": {str(i): v for i, v in enumerate(
            ["film", "historia", "jedzenie", "medycyna",
             "motoryzacja", "praca", "sport", "technologie"])},
        "template": "Tekst: {text}\nTemat tego tekstu to",
        "choices": ["film", "historia", "jedzenie", "medycyna",
                    "motoryzacja", "praca", "sport", "technologie"],
    },
}


def load_task(name, n, seed, max_bytes):
    from datasets import load_dataset
    cfg = TASKS[name]
    ds = None
    last = None
    for hid in cfg["hf_ids"]:
        try:
            ds = load_dataset(hid, split=cfg["split"])
            print(f"  zbior: {hid} ({len(ds)} pozycji)")
            break
        except Exception as e:
            last = e
    if ds is None:
        raise SystemExit(f"Nie udalo sie wczytac zbioru dla '{name}'. "
                         f"Sprobowane: {cfg['hf_ids']}\nOstatni blad: {str(last)[:200]}")

    cols = ds.column_names
    tf = next((f for f in cfg["text_field"] if f in cols), None)
    lf = next((f for f in cfg["label_field"] if f in cols), None)
    if tf is None or lf is None:
        raise SystemExit(f"Nie znalazlem kolumn. Dostepne: {cols}")

    # jesli zbior deklaruje ClassLabel, bierzemy nazwy stamtad
    choices = list(cfg["choices"])
    labmap = dict(cfg["labels"])
    feat = ds.features.get(lf)
    names = getattr(feat, "names", None)
    if names:
        missing = [n for n in names if n not in EN2PL]
        if missing:
            raise SystemExit(f"Brak tlumaczenia dla klas: {missing}. Uzupelnij EN2PL.")
        choices = [EN2PL[n] for n in names]
        labmap = {str(i): EN2PL[n] for i, n in enumerate(names)}
        labmap.update({n: EN2PL[n] for n in names})
        print(f"  klasy ze zbioru: {list(zip(range(len(names)), choices))}")

    items = []
    for row in ds:
        raw = str(row[lf])
        gold = labmap.get(raw)
        if gold is None or gold not in choices:
            continue
        txt = row[tf]
        b = txt.encode("utf-8")
        if len(b) > max_bytes:
            txt = b[:max_bytes].decode("utf-8", errors="ignore")
        items.append((cfg["template"].format(text=txt), gold))
    if not items:
        raise SystemExit("Zero pozycji po mapowaniu etykiet — sprawdz labmap/choices.")
    random.Random(seed).shuffle(items)
    return (items[:n] if n else items), choices


# ---------------------------------------------------------------- modele

def load_nanogpt(ckpt_path, device):
    from model import GPT, GPTConfig
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
    model.eval().to(device)
    return model, ck["model_args"]["block_size"]


def load_hf(name, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name).eval().to(device)
    return m, getattr(m.config, "n_positions", 1024)


def get_tokenizer(spec):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(spec) if os.path.exists(spec) else Tokenizer.from_pretrained(spec)
    eot = tok.token_to_id("<|endoftext|>")
    return tok, (eot if eot is not None else tok.get_vocab_size() - 1)


@torch.no_grad()
def choice_logprob(model, tok, eot, prompt, choice, block, device, is_hf):
    """Suma log P(tokeny odpowiedzi | prompt) oraz ta sama suma / bajty odpowiedzi."""
    cont = " " + choice
    p_ids = tok.encode(prompt).ids
    c_ids = tok.encode(cont).ids
    ids = [eot] + p_ids + c_ids
    if len(ids) > block:                      # przytnij POCZATEK promptu, nie odpowiedz
        ids = ids[-block:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    if is_hf:
        logits = model(x).logits[0, :-1]
    else:
        logits = model(x[:, :-1], x[:, 1:])[0][0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = x[0, 1:]
    n = len(c_ids)
    sel = lp[-n:].gather(1, tgt[-n:].unsqueeze(1)).sum().item()
    return sel, sel / max(len(cont.encode("utf-8")), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--ckpt")
    ap.add_argument("--hf")
    ap.add_argument("--tokenizer")
    ap.add_argument("--label", default="run")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bytes", type=int, default=1200)
    ap.add_argument("--out-dir", default="bench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.report:
        rows = [json.load(open(q)) for p in a.report for q in glob.glob(p)]
        if not rows:
            return print("brak wynikow")
        rows.sort(key=lambda r: (r["task"], -r["acc"]))
        w = max(len(r["label"]) for r in rows)
        print(f"\n{'zadanie':<10} {'model':<{w}} {'acc':>7} {'acc/bajt':>9} {'acc PMI':>9} "
              f"{'n':>5} {'losowo':>7} {'wiekszosc':>10}")
        print("-" * (56 + w))
        for r in rows:
            mb = r.get('majority_baseline')
            mbs = f"{mb*100:>9.1f}%" if mb else f"{'—':>10}"
            print(f"{r['task']:<10} {r['label']:<{w}} {r['acc']*100:>6.1f}% "
                  f"{r['acc_norm']*100:>8.1f}% {r.get('acc_pmi',0)*100:>8.1f}% "
                  f"{r['n']:>5} {100/r['n_choices']:>6.1f}% {mbs}")
        print("\nPMI = najuczciwsza miedzy tokenizerami (usuwa obciazenie dlugoscia etykiety).")
        return

    if not a.task or not (a.ckpt or a.hf):
        ap.error("podaj --task oraz --ckpt lub --hf (albo --report)")

    print(f"zadanie: {a.task}")
    items, choices = load_task(a.task, a.n, a.seed, a.max_bytes)
    import collections as _c
    dist = _c.Counter(g for _, g in items)
    maj_lab, maj_n = dist.most_common(1)[0]
    majority = maj_n / len(items)
    print(f"  pozycji do oceny: {len(items)}, klas: {len(choices)}")
    print(f"  rozklad klas: {dict(dist)}")
    print(f"  LINIA BAZOWA klasy wiekszosciowej ({maj_lab}): {majority*100:.1f}% "
          f"— to jest prog, nie {100/len(choices):.1f}%")

    if a.ckpt:
        if not a.tokenizer:
            ap.error("--ckpt wymaga --tokenizer")
        model, block = load_nanogpt(a.ckpt, a.device)
        tok, eot = get_tokenizer(a.tokenizer); is_hf = False
    else:
        model, block = load_hf(a.hf, a.device)
        tok, eot = get_tokenizer(a.tokenizer or a.hf); is_hf = True

    # bazowy logprob kazdej etykiety BEZ tresci zadania — do normalizacji PMI.
    # Usuwa obciazenie wynikajace z tego, ze etykiety maja rozna dlugosc tokenowa
    # i rozna czestosc w korpusie. To wazne przy porownywaniu ROZNYCH tokenizerow:
    # "pozytywny" to 1 token u tokenizera PL, ale kilka u GPT-2.
    # PMI WARUNKOWA DOMENOWO (Holtzman i in. 2021, "Surface Form Competition").
    # Baza to TEN SAM szablon z pusta trescia — nie neutralne "Odpowiedz:".
    # Kontekst spoza domeny daje bledna korekte: wyniki spadaja PONIZEJ poziomu
    # losowego, bo odejmujemy rozklad, ktory nie ma zwiazku z zadaniem.
    NEUTRAL = TASKS[a.task]["template"].format(text="").strip()
    base = {c: choice_logprob(model, tok, eot, NEUTRAL, c, block, a.device, is_hf)[0]
            for c in choices}
    print(f"  kontekst bazowy PMI: {NEUTRAL!r}")
    print("  bazowe logprob etykiet (im blizej siebie, tym mniejsze obciazenie):")
    for c, v in base.items():
        print(f"    {c:<18} {v:8.3f}  ({len(tok.encode(' '+c).ids)} tok.)")

    ok = ok_norm = ok_pmi = 0
    per_item = []
    for i, (prompt, gold) in enumerate(items):
        scores = [choice_logprob(model, tok, eot, prompt, c, block, a.device, is_hf)
                  for c in choices]
        pred     = choices[max(range(len(choices)), key=lambda k: scores[k][0])]
        pred_n   = choices[max(range(len(choices)), key=lambda k: scores[k][1])]
        pred_pmi = choices[max(range(len(choices)),
                               key=lambda k: scores[k][0] - base[choices[k]])]
        ok += int(pred == gold); ok_norm += int(pred_n == gold); ok_pmi += int(pred_pmi == gold)
        per_item.append([i, int(pred == gold), int(pred_n == gold), int(pred_pmi == gold)])
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(items)}  acc={ok/(i+1)*100:.1f}%  "
                  f"pmi={ok_pmi/(i+1)*100:.1f}%", flush=True)

    res = {"label": a.label, "task": a.task, "model": a.ckpt or a.hf,
           "tokenizer": a.tokenizer or a.hf, "n": len(items), "n_choices": len(choices),
           "acc": ok / len(items), "acc_norm": ok_norm / len(items),
           "acc_pmi": ok_pmi / len(items),
           "random_baseline": 1 / len(choices), "majority_baseline": majority,
           "majority_label": maj_lab, "max_bytes": a.max_bytes,
           "per_item": per_item}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.task}_{a.label}.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n{a.label} / {a.task}:  acc = {res['acc']*100:.1f}%   "
          f"acc(bajt) = {res['acc_norm']*100:.1f}%   acc(PMI) = {res['acc_pmi']*100:.1f}%   "
          f"(losowo {res['random_baseline']*100:.1f}%, wiekszosciowa {majority*100:.1f}%)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
