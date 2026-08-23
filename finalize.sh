#!/usr/bin/env bash
# Komplet pomiarow koncowych: 6 modeli x held-out, tabela, bootstrap, statystyki ziaren.
#
#   bash finalize.sh
#
# Wymaga zakonczonych wszystkich przebiegow. Wyniki laduja w results/.
set -e

echo "=== sprawdzam, czy wszystkie checkpointy istnieja ==="
MISSING=0
for d in r1-tierB-pltok r1-s1338 r1-s1339 r2-tierB-gpt2tok r2a-s1338 r2a-s1339 r2b-tierB-gpt2tok-eqtok; do
  if [ -f "out/$d/ckpt.pt" ]; then echo "  OK   $d"; else echo "  BRAK $d"; MISSING=1; fi
done
[ $MISSING -eq 1 ] && { echo "Przerywam — brakuje checkpointow."; exit 1; }

echo
echo "=== archiwizuje stare wyniki ==="
[ -d results ] && mv results "results_old_$(date +%H%M%S)"
mkdir -p results

echo
echo "=== 1/3  BPB na held-oucie (7 modeli) ==="
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r1-tierB-pltok/ckpt.pt          --tokenizer tokenizer.json --label R1-s1337
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r1-s1338/ckpt.pt                --tokenizer tokenizer.json --label R1-s1338
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r1-s1339/ckpt.pt                --tokenizer tokenizer.json --label R1-s1339
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2-tierB-gpt2tok/ckpt.pt        --tokenizer gpt2 --label R2a-s1337
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2a-s1338/ckpt.pt               --tokenizer gpt2 --label R2a-s1338
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2a-s1339/ckpt.pt               --tokenizer gpt2 --label R2a-s1339
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2b-tierB-gpt2tok-eqtok/ckpt.pt --tokenizer gpt2 --label R2b-s1337
python eval_bpb.py --eval data/heldout.jsonl --hf gpt2 --label R3-gpt2-zeroshot

echo
echo "=== KONTROLA: pole 'bytes' musi byc identyczne we wszystkich plikach ==="
grep -h '"bytes"' results/*.json | grep -v perdoc | sort -u
echo "(jesli powyzej jest wiecej niz JEDNA linia — porownanie jest niewazne)"

echo
echo "=== 2/3  tabela zbiorcza ==="
python eval_bpb.py --report 'results/*.json'

echo
echo "=== 3/3  wariancja ziarna vs wielkosc efektu ==="
python - <<'EOF'
import json, glob, statistics as st
g = {}
for p in glob.glob('results/*.json'):
    if 'perdoc' in p: continue
    r = json.load(open(p))
    g.setdefault(r['label'].rsplit('-s', 1)[0], []).append(r['bpb'])

print(f"\n{'model':<8} {'n':>2} {'srednia':>9} {'sd':>8} {'wartosci'}")
print('-'*60)
stats = {}
for k, v in sorted(g.items()):
    sd = st.stdev(v) if len(v) > 1 else 0.0
    stats[k] = (st.mean(v), sd, len(v))
    print(f"{k:<8} {len(v):>2} {st.mean(v):>9.4f} {sd:>8.4f}  {[round(x,4) for x in v]}")

if 'R1' in stats and 'R2a' in stats:
    m1, s1, n1 = stats['R1']; m2, s2, n2 = stats['R2a']
    pooled = ((s1**2 + s2**2)/2) ** 0.5 if n1 > 1 and n2 > 1 else max(s1, s2)
    d = m2 - m1
    print(f"\nR2a - R1 = {d:+.4f}   sd lacznie = {pooled:.4f}   "
          f"= {abs(d)/pooled:.1f} odchylen" if pooled else "")
    print(f"  -> {'R2a lepszy' if d < 0 else 'R1 lepszy'} o {abs(d)/m1*100:.1f}%")
if 'R1' in stats and 'R2b' in stats:
    m1, s1, _ = stats['R1']; m2, _, _ = stats['R2b']
    if s1:
        print(f"\nR2b - R1 = {m2-m1:+.4f}   = {abs(m2-m1)/s1:.1f} odchylen (sd R1)")
        print(f"  -> R1 lepszy o {(m2-m1)/m1*100:.1f}%")
EOF

echo
echo "=== bootstrap po dokumentach (uzupelnienie, nie zamiennik) ==="
python bootstrap_bpb.py results/*_perdoc.json --vs R1-s1337

echo
echo "GOTOWE. Wyniki w results/"
