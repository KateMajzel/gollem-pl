#!/usr/bin/env python3
"""Sparowany bootstrap: czy roznica BPB miedzy dwoma modelami jest realna?

Roznica 1% na jednym ziarnie losowym nie upowaznia do zadnego wniosku.
Ten skrypt szacuje przedzial ufnosci resamplingiem po dokumentach held-outu
— bez ponownego treningu.

SPAROWANY, bo oba modele oceniano na TYCH SAMYCH dokumentach: w kazdej
iteracji losujemy jeden zestaw indeksow i liczymy BPB obu modeli na nim.
To usuwa wariancje wynikajaca z tego, ze niektore dokumenty sa po prostu
trudniejsze — a zostaje wariancja samej roznicy, czyli to, co mierzymy.

    python bootstrap_bpb.py results/R1-pltok_perdoc.json results/R2a-gpt2tok_perdoc.json
    python bootstrap_bpb.py results/*_perdoc.json --vs R1-pltok      # wszystkie vs R1

Uwaga metodologiczna: to mowi tylko, czy roznica jest stabilna WZGLEDEM
DOBORU DOKUMENTOW. Nie obejmuje wariancji z ziarna treningowego — na to
trzeba powtorzyc trening z innym seedem (3-5 razy), co jest drozsze.
Zaraportuj to ograniczenie.
"""
import argparse, json, math, os, random, sys


def load(path):
    d = json.load(open(path))
    per = {int(i): (nll, b, t) for i, nll, b, t in d["per_doc"]}
    return d["label"], per


def bpb(idxs, per):
    nll = sum(per[i][0] for i in idxs)
    byt = sum(per[i][1] for i in idxs)
    return nll / (math.log(2) * byt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="pliki *_perdoc.json")
    ap.add_argument("--vs", default=None, help="etykieta modelu odniesienia (domyslnie pierwszy)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=95.0)
    a = ap.parse_args()

    runs = {}
    for f in a.files:
        if not os.path.exists(f):
            print(f"pomijam (brak): {f}"); continue
        lab, per = load(f)
        runs[lab] = per
    if len(runs) < 2:
        raise SystemExit("Potrzebne co najmniej dwa pliki *_perdoc.json")

    # tylko dokumenty obecne we WSZYSTKICH przebiegach
    common = set.intersection(*(set(p) for p in runs.values()))
    common = sorted(common)
    print(f"dokumentow wspolnych: {len(common)}")
    for lab, per in runs.items():
        if len(per) != len(common):
            print(f"  uwaga: {lab} mial {len(per)} dok., uzywam {len(common)}")

    base = a.vs or list(runs)[0]
    if base not in runs:
        raise SystemExit(f"Brak modelu odniesienia '{base}'. Dostepne: {list(runs)}")

    print(f"\nBPB na pelnym held-oucie ({len(common)} dok.):")
    full = {lab: bpb(common, per) for lab, per in runs.items()}
    for lab, v in sorted(full.items(), key=lambda kv: kv[1]):
        print(f"  {lab:<24} {v:.4f}")

    # bootstrap
    rng = random.Random(a.seed)
    N = len(common)
    others = [l for l in runs if l != base]
    diffs = {l: [] for l in others}
    for _ in range(a.n_boot):
        sample = [common[rng.randrange(N)] for _ in range(N)]
        b0 = bpb(sample, runs[base])
        for l in others:
            diffs[l].append(bpb(sample, runs[l]) - b0)

    lo_q, hi_q = (100 - a.ci) / 2 / 100, 1 - (100 - a.ci) / 2 / 100
    print(f"\nRoznica BPB wzgledem {base}  (dodatnia = gorszy od {base})")
    print(f"bootstrap sparowany, B={a.n_boot}, CI {a.ci:.0f}%\n")
    print(f"{'model':<24} {'delta':>9} {'CI dolne':>10} {'CI gorne':>10}  wniosek")
    print("-" * 74)
    for l in others:
        d = sorted(diffs[l])
        point = full[l] - full[base]
        lo, hi = d[int(lo_q * len(d))], d[int(hi_q * len(d))]
        if lo > 0:
            verdict = f"{l} ISTOTNIE gorszy"
        elif hi < 0:
            verdict = f"{l} ISTOTNIE lepszy"
            verdict = f"{l} ISTOTNIE lepszy"
        else:
            verdict = "BRAK istotnej roznicy (CI obejmuje 0)"
        print(f"{l:<24} {point:>+9.4f} {lo:>+10.4f} {hi:>+10.4f}  {verdict}")

    print("\nUwaga: bootstrap obejmuje wariancje doboru dokumentow, NIE ziarna")
    print("treningowego. Na to trzeba powtorzyc trening 3-5x z roznymi seedami.")


if __name__ == "__main__":
    main()
