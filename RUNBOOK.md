# RUNBOOK — trening własnego modelu PL lokalnie (RTX 5080, 16 GB)

Kolejność jest istotna. Każdy krok ma warunek „czy zadziałało" — nie przechodź dalej,
jeśli nie jest spełniony.

---

## 0. Katalog roboczy

```bash
mkdir -p ~/golem-projekt && cd ~/golem-projekt
df -h .        # potrzeba ~80 GB wolnego na SSD (NVMe, nie HDD)
```

Terminal: **WSL2 Ubuntu albo natywny Linux**. Nie PowerShell — `torch.compile` tam nie działa.

---

## 1. Środowisko

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git tmux
python3 -m venv .venv
source .venv/bin/activate          # w KAŻDYM nowym terminalu

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy tokenizers transformers datasets tqdm speakleash huggingface_hub
```

**Warunek przejścia:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 2.7+ / True / (12, 0)
```

`(12, 0)` = Blackwell sm_120. Build z CUDA 12.4 NIE ma kerneli dla tej karty.

---

## 2. Kod

```bash
git clone https://github.com/karpathy/nanoGPT.git golem
cd golem
rm -rf .git data/openwebtext data/shakespeare data/shakespeare_char
git init && git add -A && git commit -m "start: nanoGPT jako baza"
```

Skopiuj do `golem/`: `download_data.py`, `prepare_data.py`, `eval_bpb.py`,
`export_to_hf.py`, `probe_vram.py`, `configs/`, oraz własny `tokenizer.json`.

W README dopisz: „bazuje na nanoGPT (MIT, karpathy), commit `<hash>`" — MIT tego wymaga.

---

## 3. Trzy patche w `train.py`

### a) meta.json zamiast meta.pkl (~linia 155)

```python
meta_path = os.path.join(data_dir, 'meta.json')
if os.path.exists(meta_path):
    import json
    with open(meta_path) as f: meta = json.load(f)
    meta_vocab_size = meta['vocab_size']
    bytes_per_token = meta['bytes_per_token']['train']
    print(f"vocab_size = {meta_vocab_size}, bytes_per_token = {bytes_per_token:.3f}")
else:
    bytes_per_token = 0.0
```

### b) logowanie przebytych bajtów

```python
gb = iter_num * tokens_per_iter * bytes_per_token / 1e9
print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.0f}ms, mfu {running_mfu*100:.2f}%, {gb:.2f} GB")
```

### c) checkpoint awaryjny co 250 iteracji

```python
if iter_num % 250 == 0 and iter_num > 0 and master_process:
    torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_args': model_args, 'iter_num': iter_num, 'config': config},
               os.path.join(out_dir, 'ckpt_last.pt'))
```

Bez (a) nie ruszy. Bez (b) nie zmierzysz budżetu w bajtach. Bez (c) crash w 8. godzinie boli.

```bash
git add -A && git commit -m "patche: meta.json, logowanie bajtow, ckpt_last"
```

---

## 4. Dane

```bash
python download_data.py --list
python download_data.py --target-gb 3 --out data/raw --heldout data/heldout.jsonl
```

20–60 min. Held-out (2000 dok.) jest wycinany PRZED zapisem treningu.

**Warunek przejścia:** `ls data/raw/*.jsonl.gz` pokazuje shardy, `wc -l data/heldout.jsonl` ≈ 2000.

---

## 5. Tokenizacja (oba tokenizery, ten sam korpus)

```bash
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer tokenizer.json --out data/pl_pltok  --dedup
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer gpt2           --out data/pl_gpt2tok --dedup
```

**Zapisz `bytes_per_token` z obu** — potrzebne w kroku 7. Oczekiwane: ~5,3 (PL) i ~2,3 (GPT-2).

---

## 6. Pomiar karty

```bash
nvidia-smi --query-gpu=memory.used --format=csv     # baseline pulpitu
python probe_vram.py --n-layer 8 --n-embd 512 --vocab 32768
```

Użyj `batch_size` i `gradient_accumulation_steps`, które poda skrypt — nie moich z configów,
jeśli się różnią. Zostaw ~1 GB zapasu ponad to, co pokaże jako „OK".

---

## 7. Smoke test — 15 minut, NIE POMIJAJ

```bash
python train.py configs/golem_45m_tierB.py --max_iters=250 --eval_interval=50 --out_dir=out/smoke
python sample.py --out_dir=out/smoke --start="Polska jest krajem"
```

**Warunek przejścia:**
- loss startuje ~10,4 (= ln 32768) i po 250 krokach jest < 7,0
- `sample.py` wypluwa polski bełkot Z POLSKIMI ZNAKAMI (nie krzaki)

Krzaki = `sample.py` nadal używa tokenizera GPT-2, podmień go na własny.
Loss w miejscu lub `nan` = problem z danymi albo LR. Nie idź dalej.

---

## 8. Właściwy trening

Przelicz iteracje z liczby z kroku 5:

```
max_iters = TARGET_BYTES / bytes_per_token / (batch_size × grad_accum × block_size)
```

```bash
tmux new -s golem
source ../.venv/bin/activate
python train.py configs/golem_45m_tierB.py 2>&1 | tee log_r1.txt
# Ctrl+B, D  -> odłącz;   tmux attach -t golem  -> wróć
```

Po godzinie, w drugim terminalu:

```bash
nvidia-smi dmon -s pct -c 20      # spadek zegara >20% = throttling, dolicz +20-30% czasu
```

Potem ablacja: `python train.py configs/golem_45m_tierB_gpt2tok.py 2>&1 | tee log_r2.txt`

Higiena długiego runu:
- `powercfg /change standby-timeout-ac 0` (Windows) + wstrzymaj Windows Update
- zamknięcie okna WSL ubija proces — stąd tmux
- `export HF_HOME=/duzy/dysk/hf` żeby cache nie zapchał dysku systemowego

---

## 9. Porównanie (BPB, nie perplexity)

```bash
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r1-tierB-pltok/ckpt.pt   --tokenizer tokenizer.json --label R1-pltok
python eval_bpb.py --eval data/heldout.jsonl --ckpt out/r2-tierB-gpt2tok/ckpt.pt --tokenizer gpt2 --label R2-gpt2tok
python eval_bpb.py --eval data/heldout.jsonl --hf gpt2 --label R3-gpt2-zeroshot
python eval_bpb.py --report 'results/*.json'
```

R1 vs R2 = czysty efekt tokenizera (jedyne uczciwe porównanie).
R1 vs R3 wygląda spektakularnie i niczego nie dowodzi — napisz to wprost w raporcie.

---

## 10. Publikacja

```bash
python export_to_hf.py --ckpt out/r1-pltok/ckpt.pt --tokenizer tokenizer.json --out hf/GoLLeM-110M-PL-v2
huggingface-cli login
huggingface-cli upload TWOJ_LOGIN/GoLLeM-110M-PL-v2 hf/GoLLeM-110M-PL-v2 .
```

Karta modelu (`hf/.../README.md`) musi zawierać: miks danych z licencjami źródeł,
budżet treningu (bajty + tokeny + FLOPs), tabelę BPB z R1–R3, ograniczenia
(model 110M nie nadaje się do produkcji), i notę o nanoGPT/MIT.

---

## Szybka diagnostyka

| objaw | przyczyna |
|---|---|
| `no kernel image is available` | torch bez CUDA 12.8, sm_120 nieobsługiwane |
| OOM w 3000. iteracji (nie na starcie) | fragmentacja + przeglądarka zjadła VRAM; zmniejsz batch o 1 |
| loss = `nan` po kilkudziesięciu krokach | LR za wysoki albo uszkodzony shard; sprawdź `grad_clip=1.0` |
| loss stoi na ~10,4 | zły `vocab_size` w configu vs meta.json |
| `sample.py` daje krzaki | tokenizer GPT-2 zaszyty w `sample.py` |
| trening 3× wolniejszy niż z probe | `compile=False` albo throttling termiczny |
| dysk pełny w połowie tokenizacji | cache HF/SpeakLeash na dysku systemowym |
