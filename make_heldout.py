#!/usr/bin/env python3
"""Wycina wlasciwy held-out z pobranych shardow i USUWA go z korpusu treningowego.

Potrzebne, bo download_data.py zle oszacowal interwal probkowania i odlozyl
za malo dokumentow. Ten skrypt dobiera brakujace, stratyfikujac po zrodle,
i przepisuje shardy bez wybranych dokumentow.

    # podglad — nic nie zmienia:
    python make_heldout.py --raw data/raw --heldout data/heldout.jsonl --target 2000

    # wykonanie:
    python make_heldout.py --raw data/raw --heldout data/heldout.jsonl --target 2000 --apply

Wybor jest deterministyczny (hash tresci + seed), wiec powtarzalny.
Stare shardy laduja w data/raw_backup/ — mozna skasowac po weryfikacji.
"""
import argparse, collections, gzip, hashlib, json, os, shutil, sys


def doc_hash(text, seed):
    return int(hashlib.sha1((str(seed) + text[:2000]).encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--heldout", default="data/heldout.jsonl")
    ap.add_argument("--target", type=int, default=2000, help="docelowa laczna liczba dokumentow held-out")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-chars", type=int, default=500, help="held-out tylko z sensownie dlugich dokumentow")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    shards = sorted(f for f in os.listdir(a.raw) if f.endswith(".jsonl.gz"))
    if not shards:
        raise SystemExit(f"Brak shardow w {a.raw}")

    # ile juz mamy i z jakich zrodel
    existing, have_src = 0, collections.Counter()
    if os.path.exists(a.heldout):
        with open(a.heldout, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing += 1
                    have_src[json.loads(line).get("source", "?")] += 1
    need = max(0, a.target - existing)
    print(f"held-out obecnie: {existing} dok.  cel: {a.target}  do dobrania: {need}")
    if need == 0:
        print("Nic do zrobienia.")
        return

    # PRZEBIEG 1: policz dokumenty per zrodlo, zeby stratyfikowac
    print("\nprzebieg 1/2 — liczenie dokumentow per zrodlo...")
    per_src = collections.Counter()
    for sh in shards:
        with gzip.open(os.path.join(a.raw, sh), "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if len(d.get("text", "")) >= a.min_chars:
                        per_src[d.get("source", "?")] += 1
    total = sum(per_src.values())
    print(f"  kandydatow: {total:,} dok. z {len(per_src)} zrodel")

    # ile wziac z kazdego zrodla — proporcjonalnie
    quota = {}
    for src, n in per_src.items():
        quota[src] = max(1, round(need * n / total))
    print("\nkwoty per zrodlo:")
    for src, q in sorted(quota.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<32} {q:>5}  (juz mam {have_src.get(src,0)})")

    # PRZEBIEG 2: wybor deterministyczny — z kazdego zrodla bierzemy dokumenty
    # o najmniejszym hashu, co jest rownowazne losowaniu bez zwracania
    print("\nprzebieg 2/2 — wybor i przepisanie shardow...")
    cands = collections.defaultdict(list)   # src -> [(hash, shard, lineno)]
    for sh in shards:
        with gzip.open(os.path.join(a.raw, sh), "rt", encoding="utf-8") as f:
            for ln, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                t = d.get("text", "")
                if len(t) < a.min_chars:
                    continue
                cands[d.get("source", "?")].append((doc_hash(t, a.seed), sh, ln))

    chosen = set()
    for src, q in quota.items():
        picks = sorted(cands[src])[:q]
        for _, sh, ln in picks:
            chosen.add((sh, ln))
    print(f"  wybrano {len(chosen)} dokumentow")

    if not a.apply:
        print(f"\nDRY-RUN — nic nie zapisano.")
        print(f"Po wykonaniu held-out mialby {existing + len(chosen)} dokumentow.")
        print("Zeby wykonac, dodaj --apply")
        return

    backup = a.raw.rstrip("/") + "_backup"
    os.makedirs(backup, exist_ok=True)
    added, removed = 0, 0
    with open(a.heldout, "a", encoding="utf-8") as ho:
        for sh in shards:
            src_path = os.path.join(a.raw, sh)
            tmp_path = src_path + ".tmp"
            with gzip.open(src_path, "rt", encoding="utf-8") as fin, \
                 gzip.open(tmp_path, "wt", encoding="utf-8") as fout:
                for ln, line in enumerate(fin):
                    if not line.strip():
                        continue
                    if (sh, ln) in chosen:
                        ho.write(line)          # do held-outu
                        added += 1
                        removed += 1
                    else:
                        fout.write(line)        # zostaje w treningu
            shutil.move(src_path, os.path.join(backup, sh))
            shutil.move(tmp_path, src_path)
            print(f"  {sh} przepisany", flush=True)

    print(f"\ndodano do held-outu: {added}   usunieto z treningu: {removed}")
    print(f"held-out lacznie:    {existing + added} dokumentow")
    print(f"kopia starych shardow: {backup}/  (mozna skasowac po weryfikacji)")
    print("\nsprawdz rozmiary:")
    print(f"  du -sh {a.raw} {backup}")
    print(f"  wc -l {a.heldout}")


if __name__ == "__main__":
    main()
