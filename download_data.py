#!/usr/bin/env python3
"""Pobiera polski korpus przez SpeakLeash i zapisuje jako .jsonl.gz + held-out.

    pip install speakleash

    python download_data.py --list
    python download_data.py --target-gb 3 --out data/raw --heldout data/heldout.jsonl

Held-out jest wycinany PRZED zapisem treningu (co N-ty dokument), wiec trening
fizycznie go nie widzi. To warunek sensownosci ewaluacji BPB.

UWAGA: SpeakLeash replikuje archiwa na dysk (katalog --cache). Licz sie z tym,
ze zajmie to tyle samo miejsca co wynikowe .jsonl.gz, a czesto wiecej.
"""
import argparse, gzip, json, os, random, sys

# Domyslny miks — nazwy zweryfikowane wzgledem katalogu SpeakLeash (--list).
# Wagi = docelowy udzial w korpusie (sumuja sie do 1.0).
#
# CELOWO POMINIETE:
#   text_translated, web_synt_*  -> translationese i tekst syntetyczny
#   tekstowo_*                   -> teksty piosenek (prawa autorskie)
#   open_subtitles_corpus        -> napisy, prawie zawsze tlumaczone z EN
#   cc_*, *_madlad, HPLT_1xx+    -> surowy Common Crawl bez filtracji jakosci
#   eurlex/saos/CBOSA            -> prawo; tokenizer juz jest w to przechylony
DEFAULT_MIX = [
    # --- wysoka jakosc: encyklopedia i literatura (37%) ---
    ("plwiki",                        0.20),   # 1797 MB — fakty, poprawna skladnia
    ("plwikisource",                  0.08),   # 1883 MB — teksty zrodlowe, starsza polszczyzna
    ("wolne_lektury_corpus",          0.05),   #  250 MB — literatura, domena publiczna
    ("1000_novels_corpus_CLARIN-PL",  0.04),   #  145 MB — proza wspolczesna

    # --- jezyk zywy: fora (22%) ---
    ("forum_kafeteria_pl_corpus",     0.07),   # jezyk potoczny, dialog
    ("forum_gazeta_pl_corpus",        0.07),
    ("forum_elektroda_pl_corpus",     0.05),   # techniczny, ale naturalny
    ("forum_trojmiasto_pl_corpus",    0.03),

    # --- publicystyka i nauka (11%) ---
    ("news_3_automotive_corpus",      0.02),
    ("news_4_business_corpus",        0.02),
    ("news_5_lifestyle_corpus",       0.02),
    ("biblioteka_nauki_pl_corpus",    0.05),   # jezyk naukowy

    # --- urzedowy: MALO, tokenizer juz to nadreprezentuje (3%) ---
    ("ISAP_corpus",                   0.03),

    # --- web, umiarkowanie (27%) ---
    ("HPLT_1",                        0.09),
    ("HPLT_2",                        0.09),
    ("web_artykuły_inne_1",           0.09),
]


def pick_datasets(sl, patterns, verbose=True):
    import fnmatch
    avail = {d.name: d for d in sl.datasets}
    chosen, used = [], set()
    for pat, weight in patterns:
        hits = [n for n in sorted(avail) if fnmatch.fnmatch(n, pat) and n not in used]
        if not hits:
            if verbose:
                print(f"  (brak dopasowania dla '{pat}' - pomijam)")
            continue
        for n in hits:
            used.add(n)
        chosen.append((hits, weight))
    return chosen, avail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="wypisz dostepne zbiory i zakoncz")
    ap.add_argument("--target-gb", type=float, default=3.0, help="docelowy rozmiar korpusu (GB tekstu)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--heldout-docs", type=int, default=2000)
    ap.add_argument("--cache", default="data/sl_cache")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    try:
        from speakleash import Speakleash
    except ImportError:
        raise SystemExit("Brak pakietu: pip install speakleash")

    os.makedirs(a.cache, exist_ok=True)
    sl = Speakleash(os.path.abspath(a.cache))

    if a.list:
        rows = sorted(sl.datasets, key=lambda d: -d.characters)
        print(f"{'zbior':<28} {'MB':>9} {'dokumentow':>12}")
        print("-" * 52)
        for d in rows:
            print(f"{d.name:<28} {d.characters/1024/1024:>9,.0f} {d.documents:>12,}")
        print(f"\nRAZEM: {sum(d.characters for d in rows)/1024**3:,.1f} GB znakow")
        return

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(a.heldout) or ".", exist_ok=True)
    rng = random.Random(a.seed)

    groups, avail = pick_datasets(sl, DEFAULT_MIX)
    if not groups:
        raise SystemExit("Nie udalo sie dobrac zadnego zbioru - sprawdz --list.")

    target_bytes = a.target_gb * 1e9
    plan = []
    for names, weight in groups:
        per = target_bytes * weight / len(names)
        for n in names:
            plan.append((n, per))
    print("plan pobierania:")
    miss = [n for n, _ in plan if n not in avail]
    for n, b in plan:
        mark = "  BRAK!" if n not in avail else ""
        print(f"  {n:<32} do {b/1e6:>8,.0f} MB{mark}")
    if miss:
        print(f"\n!! {len(miss)} zbiorow nie istnieje w katalogu — korpus bedzie MNIEJSZY niz {a.target_gb} GB.")
        print("   Popraw DEFAULT_MIX w tym pliku albo sprawdz nazwy przez --list.")
        if input("   Kontynuowac mimo to? [t/N] ").strip().lower() not in ("t", "y"):
            raise SystemExit("przerwane")
    print("\nUWAGA: SpeakLeash pobiera CALE archiwa, nawet gdy bierzemy z nich wycinek.")
    print("Realny transfer bedzie wiekszy niz docelowy rozmiar korpusu.\n")

    # held-out: co ile-ty dokument odkladamy
    heldout_every = max(1, int(sum(avail[n].documents for n, _ in plan if n in avail)
                              / max(a.heldout_docs, 1)))
    ho = open(a.heldout, "w", encoding="utf-8")
    n_ho = 0

    shard_i, shard_bytes, total_bytes, total_docs = 0, 0, 0, 0
    fh = gzip.open(os.path.join(a.out, f"shard_{shard_i:04d}.jsonl.gz"), "wt", encoding="utf-8")

    for name, budget in plan:
        if name not in avail:
            print(f"  pomijam {name} (niedostepny)")
            continue
        print(f"-> {name}", flush=True)
        got = 0
        try:
            for txt in sl.get(name).data:
                if not txt or len(txt) < a.min_chars:
                    continue
                b = len(txt.encode("utf-8"))
                total_docs += 1
                if total_docs % heldout_every == 0 and n_ho < a.heldout_docs:
                    ho.write(json.dumps({"text": txt, "source": name}, ensure_ascii=False) + "\n")
                    n_ho += 1
                    continue                      # NIE trafia do treningu
                fh.write(json.dumps({"text": txt, "source": name}, ensure_ascii=False) + "\n")
                got += b; shard_bytes += b; total_bytes += b
                if shard_bytes >= a.shard_mb * 1e6:
                    fh.close(); shard_i += 1; shard_bytes = 0
                    fh = gzip.open(os.path.join(a.out, f"shard_{shard_i:04d}.jsonl.gz"), "wt", encoding="utf-8")
                if got >= budget:
                    break
        except Exception as e:
            print(f"   blad przy {name}: {str(e)[:150]}")
        print(f"   pobrano {got/1e6:,.0f} MB  (lacznie {total_bytes/1e9:.2f} GB)", flush=True)

    fh.close(); ho.close()
    print(f"\ntrening : {a.out}/  {total_bytes/1e9:.2f} GB w {shard_i+1} shardach")
    print(f"held-out: {a.heldout}  {n_ho} dokumentow (trening ich NIE widzial)")
    print("\nnastepny krok:")
    print(f"  python prepare_data.py --input '{a.out}/*.jsonl.gz' --tokenizer tokenizer.json --out data/pl_pltok --dedup")


if __name__ == "__main__":
    main()
