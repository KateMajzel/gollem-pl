# gollem-pl — a tokenizer ablation for a Polish language model

A controlled experiment: **does a dedicated Polish tokenizer produce a better language model
than the GPT-2 tokenizer, given the same training budget?**

Model: [KateMajzel/GoLLeM-45M-PL](https://huggingface.co/KateMajzel/GoLLeM-45M-PL) · Tokenizer: [KateMajzel/tokenizer-pl-32k](https://huggingface.co/KateMajzel/tokenizer-pl-32k) ·
Hardware: 1× RTX 5080 (16 GB) · 6 training runs, ~7 h total

---

## Result

**It doesn't produce a better model. It produces a comparable model 2.5× cheaper.**

Four independent measurements; none shows a significant advantage for either
tokenizer at an equal text budget — and they don't even agree on direction:

| measurement | R1 (PL tok.) vs R2a (GPT-2 tok.) | significance |
|---|---|---|
| BPB, held-out set of 1,999 docs, 3 seeds | R2a better by 1.4% | t = −2.76, df = 2.5 — borderline |
| PolEmo2 (PMI) | R1 better by 3.5 pp | CI [−1.8, +8.8] — not significant |
| 8Tags (PMI) | R1 better by 1.8 pp | CI [−3.2, +6.8] — not significant |
| private held-out (15 sentences) | R1 better by 2.6% | sample too small |

### The number that settles it

| comparison | ΔBPB |
|---|---|
| **same model (R1), different random seed** | **0.0186** |
| **different tokenizer (R1 vs R2a)** | 0.0124 |

Changing the seed moves the result more than changing the tokenizer. At this training
budget, the tokenizer is not what differentiates model quality.

### Where the advantage lies

| run | tokenizer | budget | BPB (mean of 3 seeds) | sd | parameters | time |
|---|---|---|---|---|---|---|
| **R1** | Polish, 32,768 | 2.96 GB | 1.2114 | 0.0100 | 42.5M | **49.7 min** |
| R2a | GPT-2, 50,257 | 2.96 GB — same bytes | 1.1946 | 0.0035 | 51.5M | 122.4 min |
| R2b | GPT-2, 50,257 | 2,970 steps — same token count | 1.2756 | (n=1) | 51.5M | 62.4 min |
| R3 | GPT-2 pretrained, zero-shot | — | 2.9555 | (n=1) | 124M | — |

**R1 vs R2b: +5.30% in R1's favor, i.e. 6.4 standard deviations.** With compute budgets
matched, the denser tokenizer's advantage is unambiguous.

All BPB measurements are on the same 2,767,440 held-out bytes.

---

## Experiment design

Three runs close off both lines of objection a critic could take:

- **R2a** — same bytes as R1. Rebuts *"R1 read more text"*.
  Costs R2a 2.2× more compute for the same text.
- **R2b** — same steps as R1. Rebuts *"R1 got more compute"*.
  Costs R2b 1.96× less text read.

The mechanism behind R2a's advantage in the first comparison: the less dense tokenizer makes
more passes through the network per byte (1.30M steps vs 0.67M on the held-out set).
Once that is equalized, the direction reverses.

**Asymmetries working against the hypothesis** (reported deliberately): R2 has 21% more
parameters (a larger embedding table) and, in R2b, more actual FLOPs per step
(the output layer computes `2·d·V`, and `V` is larger).

---

## Tokenizer

Byte-level BPE, 32,568 entries + 200 special tokens = 32,768. GPT-4-style
pre-tokenizer, no normalization, lossless round-trip, 27.9% of entries contain Polish
diacritics, zero tokens unreachable via merges.

| | this tokenizer | GPT-2 |
|---|---|---|
| bytes/token (corpus) | 4.050 | 2.066 |
| bytes/token (held-out) | 4.103 | 2.134 |
| tokens from 2.96 GB | 731M | 1,434M |
| **density** | **1.96×** | 1.00× |

It shows in the benchmark labels too: `niejednoznaczny` ("ambiguous") is 3 tokens for us and 9 for GPT-2.

Note: fertility on polished prose is 5.80 characters/token, but on the real
corpus it is 4.05 bytes/token. Fertility should be reported on the corpus, not on hand-picked sentences.

## Data

2.96 GB, 803,177 documents after deduplication, from [SpeakLeash](https://speakleash.org/).

| category | share | datasets |
|---|---|---|
| encyclopedia, literature | 37% | `plwiki`, `plwikisource`, `wolne_lektury_corpus`, `1000_novels_corpus_CLARIN-PL` |
| forums | 22% | `kafeteria`, `gazeta`, `elektroda`, `trojmiasto` |
| journalism, science | 11% | `news_3/4/5`, `biblioteka_nauki_pl_corpus` |
| official / legal | 3% | `ISAP_corpus` |
| web | 27% | `HPLT_1`, `HPLT_2`, `web_artykuły_inne_1` |

**Deliberately excluded:** `text_translated`, `open_subtitles` (translationese),
`web_synt_*` (synthetic text), `tekstowo_*` (copyright), `cc_*`, `*_madlad`
(raw Common Crawl). Official texts were capped at 3% because the tokenizer was already
skewed in that direction.

## Benchmarks

Answer log-likelihood scoring with domain-conditional PMI normalization.

| task | R1 | R2a | GPT-2 zero-shot | majority class | random |
|---|---|---|---|---|---|
| PolEmo2-IN | **47.2%** | 43.8% | 20.8% | 40.0% | 25.0% |
| 8Tags | **31.5%** | 29.8% | 17.8% | 16.5% | 12.5% |

Both Polish-trained models clearly beat the majority class on 8Tags
(~2× the baseline). GPT-2 zero-shot is below the baseline on both tasks. The R1 vs R2a
differences are not significant (McNemar χ² = 1.51 and 0.34, against a threshold of 3.84).

**Limitation:** benchmarks were computed on a single seed per model.

---

## Five methodological pitfalls

All of them made, and fixed, during this project.

**1. Perplexity is not comparable across tokenizers.**
R1 has a per-token PPL of 30.7, R2a 5.8. That would look like a fivefold advantage for GPT-2.
In BPB the difference is 1.4%. Models with different tokenizers predict units of different
difficulty. The only valid metric is bits-per-byte.

**2. Truncating to `block_size` tokens invalidates the comparison.**
1,023 tokens is ~4,100 bytes with the Polish tokenizer and ~2,100 with GPT-2 — each
model would be scored on a different amount of text. You have to truncate by **bytes**, before tokenization.
Check: the `bytes` field must be identical across all result files. The same bug
comes back in generation: a `max_new_tokens` limit gives the models different amounts of text.

**3. A document-level bootstrap does not measure training variance.**
It produced CIs suggesting "R2a is significantly better". Reruns with other seeds showed that
the difference between seeds of the same model is larger than the measured effect. The bootstrap
answers the question *"will the difference hold on other texts"*, not *"will it hold
if you retrain"*. You need both.

**4. PMI with an out-of-domain baseline gives below-chance results.**
Normalizing as `log P(label|context) − log P(label|"Odpowiedź:")` (*"Answer:"*) gave 15–21%
against 25% chance. The baseline context must come from the same domain — the same template
with empty content. After the fix: 47.2% and 31.5%. Consistently below-chance results
always signal an implementation bug, not a weak model.

**5. nanoGPT without `targets` returns logits for the last position only.**
`model(x)[0]` has shape `(1, 1, V)`, not `(1, T, V)` — an optimization for generation.
To compute NLL you need `model(x[:, :-1], x[:, 1:])`.

Also: `always_save_checkpoint = False` does **not** protect against overwriting — a new
run starts with `best_val_loss = 1e9`, so the first evaluation treats its weak
result as an improvement and overwrites the finished model. Hence the patch that stops training when `out_dir`
already contains a checkpoint.

---

## Repository structure

```
download_data.py      download the corpus (SpeakLeash) + split off the held-out set
make_heldout.py       held-out set stratified by source + leakage audit
prepare_data.py       tokenization -> train.bin/val.bin + meta.json (bytes_per_token)
patch_nanogpt.py      4 patches to train.py and sample.py (idempotent)
probe_vram.py         measure batch_size and throughput for a given GPU
eval_bpb.py           bits-per-byte, shared byte budget, per-document NLL output
bootstrap_bpb.py      paired bootstrap confidence intervals
bench_pl.py           PolEmo2 / 8Tags, log-likelihood + domain PMI
nikos_eval.py         private held-out + side-by-side generations
export_to_hf.py       nanoGPT -> Hugging Face (Conv1D transposition, verification)
finalize.sh           full set of final measurements in one command
configs/              configs for R1, R2a, R2b
RUNBOOK.md            step-by-step procedure with gate conditions
```

## Reproduction

```bash
# 0. environment (Blackwell requires CUDA 12.8+)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy tokenizers transformers datasets speakleash huggingface_hub

# 1. base code + patches
git clone https://github.com/karpathy/nanoGPT.git && cd nanoGPT
# copy the scripts from this repo and tokenizer.json
python patch_nanogpt.py --apply

# 2. data
python download_data.py --target-gb 3 --out data/raw --heldout data/heldout.jsonl
python make_heldout.py --raw data/raw --heldout data/heldout.jsonl --target 2000 --apply

# 3. tokenization — same corpus, two tokenizers, same split
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer tokenizer.json --out data/pl_pltok  --dedup
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer gpt2           --out data/pl_gpt2tok --dedup
# read bytes_per_token from both meta.json files and recompute max_iters in the configs

# 4. training (add --seed=1338 / 1339 for reruns)
python probe_vram.py --n-layer 8 --n-head 8 --n-embd 512 --vocab 32768
python -u train.py configs/golem_45m_tierB.py               2>&1 | tee log_r1.txt
python -u train.py configs/golem_45m_tierB_gpt2tok.py       2>&1 | tee log_r2a.txt
python -u train.py configs/golem_45m_tierB_gpt2tok_eqtok.py 2>&1 | tee log_r2b.txt

# 5. full set of measurements
bash finalize.sh

# 6. export
python export_to_hf.py --ckpt out/r1-tierB-pltok/ckpt.pt --tokenizer tokenizer.json --out hf/GoLLeM-45M-PL
```

Full procedure with gate conditions: [`RUNBOOK.md`](RUNBOOK.md).

**No data or weights in the repo** — `data/`, `out/`, `hf/` are in `.gitignore`. The corpus is
rebuilt by `download_data.py`; the weights are on Hugging Face.

---

## Related work

**Bielik v3 PL** (arXiv [2604.10799](https://arxiv.org/abs/2604.10799), April 2026)
replaced the Mistral tokenizer with the Polish APT4 tokenizer in its 7B and 11B models, at a comparable
vocabulary size (~32,000). Fertility dropped from 3.22 to 1.62 tokens per word, and evaluation on
nine benchmarks showed that the models **retain** the original models' performance, outperforming
them on two.

This is the same conclusion as here — switching to a dedicated tokenizer preserves quality
and doubles representation density — reached independently, at 250× the scale.

**ATLAS** (Google Research) derived optimal scaling trajectories for six
languages, including Russian, and the curves turned out to be very similar across languages. This
suggests that the optimal tokens-per-parameter ratio does not depend strongly on a language's morphology —
although no dedicated study for Polish exists.

## Limitations

- A single scale (42M parameters, 3 GB of data) and a single architecture. Whether the findings
  transfer to larger models has not been studied.
- Three seeds for R1 and R2a, one each for R2b and the benchmarks. The standard deviation for R2b
  is borrowed from R1.
- The Chinchilla rule counts **all** parameters, embeddings included — hence 731M
  tokens / 42.5M = 17.2 tokens per parameter. With embeddings making up 40% of the model,
  we are close to the regime where naively applying the "20 tokens per parameter" rule
  is least precise.
- PolEmo2 provides morphologically segmented text (`Leczyła m się`), which differs
  from the natural Polish used in training.
- Base model, with no fine-tuning or safety filtering. It produces correct
  Polish, but hallucinates facts and loses coherence after a few sentences.
- The corpus contains Windows line endings (`\r\n`), which the model reproduces.

## License and attribution

Base code: [nanoGPT](https://github.com/karpathy/nanoGPT), MIT, Andrej Karpathy —
`LICENSE` file retained. Modifications and evaluation scripts: MIT.
Data: [SpeakLeash](https://speakleash.org/); dataset licenses as specified by each source.

## Citation

```bibtex
@misc{gollem45mpl,
  title  = {GoLLeM-45M-PL: a tokenizer ablation for a Polish language model},
  author = {Majzel-Pośpiech, Katarzyna},
  year   = {2026},
  url    = {https://github.com/KateMajzel/gollem-pl}
}
```
