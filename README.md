# gollem-pl — ablacja tokenizera dla polskiego modelu językowego

Kontrolowany eksperyment: **czy dedykowany tokenizer polski daje lepszy model językowy
niż tokenizer GPT-2, przy tym samym budżecie treningowym?**

Model: [KateMajzel/GoLLeM-45M-PL](https://huggingface.co/KateMajzel/GoLLeM-45M-PL) · Tokenizer: [KateMajzel/tokenizer-pl-32k](https://huggingface.co/KateMajzel/tokenizer-pl-32k) ·
Sprzęt: 1× RTX 5080 (16 GB) · 6 przebiegów treningowych, ~7 h łącznie

---

## Wynik

**Nie daje lepszego modelu. Daje porównywalny model 2,5× taniej.**

Cztery niezależne pomiary, żaden nie wykazuje istotnej przewagi któregokolwiek
tokenizera przy równym budżecie tekstu — a ich kierunki są niespójne:

| pomiar | R1 (tok. PL) vs R2a (tok. GPT-2) | istotność |
|---|---|---|
| BPB, held-out 1 999 dok., 3 ziarna | R2a lepszy o 1,4% | t = −2,76, df = 2,5 — na granicy |
| PolEmo2 (PMI) | R1 lepszy o 3,5 pp | CI [−1,8; +8,8] — nieistotne |
| 8Tags (PMI) | R1 lepszy o 1,8 pp | CI [−3,2; +6,8] — nieistotne |
| prywatny held-out (15 zdań) | R1 lepszy o 2,6% | próbka za mała |

### Liczba, która rozstrzyga

| porównanie | ΔBPB |
|---|---|
| **ten sam model (R1), inne ziarno losowe** | **0,0186** |
| **inny tokenizer (R1 vs R2a)** | 0,0124 |

Zmiana ziarna wpływa na wynik silniej niż zmiana tokenizera. Przy tym budżecie
treningowym tokenizer nie jest czynnikiem różnicującym jakość.

### Gdzie leży przewaga

| przebieg | tokenizer | budżet | BPB (śr. z 3 ziaren) | sd | parametry | czas |
|---|---|---|---|---|---|---|
| **R1** | polski, 32 768 | 2,96 GB | 1,2114 | 0,0100 | 42,5 M | **49,7 min** |
| R2a | GPT-2, 50 257 | 2,96 GB — te same bajty | 1,1946 | 0,0035 | 51,5 M | 122,4 min |
| R2b | GPT-2, 50 257 | 2 970 kroków — te same tokeny | 1,2756 | (n=1) | 51,5 M | 62,4 min |
| R3 | GPT-2 pretrained, zero-shot | — | 2,9555 | (n=1) | 124 M | — |

**R1 vs R2b: +5,30% na korzyść R1, czyli 6,4 odchylenia.** Przy wyrównanym budżecie
obliczeniowym przewaga gęstszego tokenizera jest jednoznaczna.

Wszystkie pomiary BPB na identycznych 2 767 440 bajtach held-outu.

---

## Projekt eksperymentu

Trzy przebiegi zamykają obie drogi ucieczki dla krytyka:

- **R2a** — te same bajty co R1. Odpiera *„R1 przeczytał więcej tekstu"*.
  Kosztuje R2a 2,2× więcej obliczeń na ten sam tekst.
- **R2b** — te same kroki co R1. Odpiera *„R1 dostał więcej obliczeń"*.
  Kosztuje R2b 1,96× mniej przeczytanego tekstu.

Mechanizm przewagi R2a w pierwszym porównaniu: rzadszy tokenizer wykonuje więcej
przejść przez sieć na ten sam bajt (1,30 mln kroków wobec 0,67 mln na held-oucie).
Gdy się to wyrówna, kierunek się odwraca.

**Asymetrie działające na niekorzyść hipotezy** (zgłaszane celowo): R2 ma 21% więcej
parametrów (większa tablica embeddingów) i w R2b więcej realnych FLOPs na krok
(warstwa wyjściowa liczy `2·d·V`, a `V` jest większe).

---

## Tokenizer

Byte-level BPE, 32 568 pozycji + 200 tokenów specjalnych = 32 768. Pre-tokenizer
w stylu GPT-4, bez normalizacji, round-trip bezstratny, 27,9% pozycji z polskimi
diakrytykami, zero tokenów nieosiągalnych przez merge.

| | ten tokenizer | GPT-2 |
|---|---|---|
| bajty/token (korpus) | 4,050 | 2,066 |
| bajty/token (held-out) | 4,103 | 2,134 |
| tokenów z 2,96 GB | 731 mln | 1 434 mln |
| **gęstość** | **1,96×** | 1,00× |

Widać to też w etykietach benchmarków: `niejednoznaczny` to 3 tokeny u nas i 9 u GPT-2.

Uwaga: fertility na dopracowanej prozie wynosi 5,80 znaku/token, ale na realnym
korpusie 4,05 bajtu/token. Fertility należy podawać z korpusu, nie z wybranych zdań.

## Dane

2,96 GB, 803 177 dokumentów po deduplikacji, [SpeakLeash](https://speakleash.org/).

| kategoria | udział | zbiory |
|---|---|---|
| encyklopedia, literatura | 37% | `plwiki`, `plwikisource`, `wolne_lektury_corpus`, `1000_novels_corpus_CLARIN-PL` |
| fora | 22% | `kafeteria`, `gazeta`, `elektroda`, `trojmiasto` |
| publicystyka, nauka | 11% | `news_3/4/5`, `biblioteka_nauki_pl_corpus` |
| urzędowe | 3% | `ISAP_corpus` |
| web | 27% | `HPLT_1`, `HPLT_2`, `web_artykuły_inne_1` |

**Celowo wykluczone:** `text_translated`, `open_subtitles` (translationese),
`web_synt_*` (tekst syntetyczny), `tekstowo_*` (prawa autorskie), `cc_*`, `*_madlad`
(surowy Common Crawl). Teksty urzędowe ograniczone do 3%, bo tokenizer był już w tę
stronę przechylony.

## Benchmarki

Metoda answer-log-likelihood z normalizacją PMI warunkową domenowo.

| zadanie | R1 | R2a | GPT-2 zero-shot | klasa większościowa | losowo |
|---|---|---|---|---|---|
| PolEmo2-IN | **47,2%** | 43,8% | 20,8% | 40,0% | 25,0% |
| 8Tags | **31,5%** | 29,8% | 17,8% | 16,5% | 12,5% |

Oba modele trenowane na polskim wyraźnie biją klasę większościową na 8Tags
(~2× próg). GPT-2 zero-shot jest poniżej progu na obu zadaniach. Różnice R1 vs R2a
nieistotne (McNemar χ² = 1,51 i 0,34 przy progu 3,84).

**Ograniczenie:** benchmarki liczone na jednym ziarnie każdego modelu.

---

## Pięć pułapek metodologicznych

Wszystkie popełnione i naprawione w trakcie tego projektu.

**1. Perplexity jest nieporównywalna między tokenizerami.**
R1 ma PPL/token 30,7, R2a — 5,8. Wyglądałoby na pięciokrotną przewagę GPT-2.
W BPB różnica wynosi 1,4%. Modele o różnych tokenizerach przewidują jednostki różnej
trudności. Jedyną poprawną metryką jest bits-per-byte.

**2. Przycinanie do `block_size` tokenów unieważnia porównanie.**
1 023 tokeny to ~4 100 bajtów przy tokenizerze polskim i ~2 100 przy GPT-2 — każdy
model oceniałby inną ilość tekstu. Trzeba przycinać po **bajtach**, przed tokenizacją.
Kontrola: pole `bytes` musi być identyczne we wszystkich plikach wyników. Ten sam błąd
powraca przy generowaniu: limit `max_new_tokens` daje modelom różną ilość tekstu.

**3. Bootstrap po dokumentach nie mierzy wariancji treningu.**
Dawał CI sugerujące „R2a istotnie lepszy". Powtórki z innymi ziarnami pokazały, że
różnica między ziarnami tego samego modelu jest większa niż mierzony efekt. Bootstrap
odpowiada na pytanie *„czy różnica utrzyma się na innych tekstach"*, nie *„czy utrzyma
się przy ponownym treningu"*. Potrzeba obu.

**4. PMI z niedomenową bazą daje wyniki poniżej poziomu losowego.**
Normalizacja `log P(etykieta|kontekst) − log P(etykieta|"Odpowiedź:")` dała 15–21%
przy losowym 25%. Kontekst bazowy musi pochodzić z tej samej domeny — ten sam szablon
z pustą treścią. Po poprawce: 47,2% i 31,5%. Konsekwentne wyniki poniżej przypadku
to zawsze sygnał błędu implementacji, nie słabości modelu.

**5. nanoGPT bez `targets` zwraca logity tylko dla ostatniej pozycji.**
`model(x)[0]` daje kształt `(1, 1, V)`, nie `(1, T, V)` — optymalizacja pod generowanie.
Do liczenia NLL trzeba `model(x[:, :-1], x[:, 1:])`.

Dodatkowo: `always_save_checkpoint = False` **nie chroni** przed nadpisaniem — nowy
przebieg startuje z `best_val_loss = 1e9`, więc pierwsza ewaluacja uznaje swój słaby
wynik za poprawę i kasuje gotowy model. Stąd patch zatrzymujący trening, gdy `out_dir`
zawiera już checkpoint.

---

## Struktura repozytorium

```
download_data.py      pobranie korpusu (SpeakLeash) + wydzielenie held-outu
make_heldout.py       held-out stratyfikowany po źródłach + audyt wycieków
prepare_data.py       tokenizacja -> train.bin/val.bin + meta.json (bytes_per_token)
patch_nanogpt.py      4 patche na train.py i sample.py (idempotentne)
probe_vram.py         pomiar batch_size i przepustowości dla danej karty
eval_bpb.py           bits-per-byte, wspólny budżet bajtów, zapis NLL per dokument
bootstrap_bpb.py      sparowany bootstrap przedziałów ufności
bench_pl.py           PolEmo2 / 8Tags, log-likelihood + PMI domenowa
nikos_eval.py         prywatny held-out + zestawienie generacji
export_to_hf.py       nanoGPT -> HuggingFace (transpozycja Conv1D, weryfikacja)
finalize.sh           komplet pomiarów końcowych jedną komendą
configs/              konfiguracje R1, R2a, R2b
RUNBOOK.md            procedura krok po kroku z warunkami przejścia
```

## Odtworzenie

```bash
# 0. środowisko (Blackwell wymaga CUDA 12.8+)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy tokenizers transformers datasets speakleash huggingface_hub

# 1. kod bazowy + patche
git clone https://github.com/karpathy/nanoGPT.git && cd nanoGPT
# skopiuj skrypty z tego repo oraz tokenizer.json
python patch_nanogpt.py --apply

# 2. dane
python download_data.py --target-gb 3 --out data/raw --heldout data/heldout.jsonl
python make_heldout.py --raw data/raw --heldout data/heldout.jsonl --target 2000 --apply

# 3. tokenizacja — ten sam korpus, dwa tokenizery, ten sam split
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer tokenizer.json --out data/pl_pltok  --dedup
python prepare_data.py --input 'data/raw/*.jsonl.gz' --tokenizer gpt2           --out data/pl_gpt2tok --dedup
# odczytaj bytes_per_token z obu meta.json i przelicz max_iters w configach

# 4. trening (dodaj --seed=1338 / 1339 dla powtórek)
python probe_vram.py --n-layer 8 --n-head 8 --n-embd 512 --vocab 32768
python -u train.py configs/golem_45m_tierB.py               2>&1 | tee log_r1.txt
python -u train.py configs/golem_45m_tierB_gpt2tok.py       2>&1 | tee log_r2a.txt
python -u train.py configs/golem_45m_tierB_gpt2tok_eqtok.py 2>&1 | tee log_r2b.txt

# 5. komplet pomiarów
bash finalize.sh

# 6. eksport
python export_to_hf.py --ckpt out/r1-tierB-pltok/ckpt.pt --tokenizer tokenizer.json --out hf/GoLLeM-45M-PL
```

Pełna procedura z warunkami przejścia: [`RUNBOOK.md`](RUNBOOK.md).

**Brak danych i wag w repo** — `data/`, `out/`, `hf/` w `.gitignore`. Korpus odtwarza
`download_data.py`, wagi są na Hugging Face.

---

## Prace pokrewne

**Bielik v3 PL** (arXiv [2604.10799](https://arxiv.org/abs/2604.10799), kwiecień 2026)
podmienił tokenizer Mistrala na polski APT4 w modelach 7B i 11B przy porównywalnym
słowniku (~32 000). Fertility spadła z 3,22 do 1,62 tokena na słowo, a ewaluacja na
dziewięciu benchmarkach wykazała, że modele **zachowują** wyniki oryginałów, na dwóch
je przewyższając.

To ten sam wniosek co tutaj — wymiana tokenizera na dedykowany zachowuje jakość
i podwaja gęstość reprezentacji — otrzymany niezależnie, na skali 250× większej.

**ATLAS** (Google Research) wyznaczył optymalne trajektorie skalowania dla sześciu
języków, w tym rosyjskiego, i krzywe okazały się między językami bardzo podobne. To
sugeruje, że optimum tokenów na parametr nie zależy silnie od morfologii języka —
choć dedykowanego studium dla polskiego brak.

## Ograniczenia

- Jedna skala (42 M parametrów, 3 GB danych), jedna architektura. Przeniesienie
  wniosków na większe modele nie jest zbadane.
- Trzy ziarna dla R1 i R2a, po jednym dla R2b i benchmarków. Odchylenie dla R2b
  pożyczone z R1.
- Reguła Chinchilli liczy **wszystkie** parametry, także embedding — stąd 731 mln
  tokenów / 42,5 M = 17,2 tokena na parametr. Przy embeddingu stanowiącym 40% modelu
  jesteśmy blisko strefy, w której proste stosowanie reguły „20 tokenów na parametr"
  jest najmniej precyzyjne.
- PolEmo2 podaje teksty po segmentacji morfologicznej (`Leczyła m się`), co odbiega
  od naturalnej polszczyzny treningowej.
- Model bazowy, bez dostrajania i filtrowania bezpieczeństwa. Produkuje poprawną
  polszczyznę, ale halucynuje fakty i traci spójność po kilku zdaniach.
- Korpus zawiera windowsowe końce linii (`\r\n`), które model odtwarza.

## Licencja i atrybucja

Kod bazowy: [nanoGPT](https://github.com/karpathy/nanoGPT), MIT, Andrej Karpathy —
plik `LICENSE` zachowany. Modyfikacje i skrypty ewaluacyjne: MIT.
Dane: [SpeakLeash](https://speakleash.org/), licencje zbiorów po stronie źródła.

## Cytowanie

```bibtex
@misc{gollem45mpl,
  title  = {GoLLeM-45M-PL: ablacja tokenizera dla polskiego modelu językowego},
  author = {Majzel-Pośpiech, Katarzyna},
  year   = {2026},
  url    = {https://github.com/KateMajzel/gollem-pl}
}
```
