---
language:
- pl
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- polish
- gpt2
- nanogpt
- tokenizer-ablation
- research
---

# GoLLeM-45M-PL

Model GPT-2 (42,5 mln parametrów) wytrenowany od zera na 2,96 GB polskiego tekstu,
z dedykowanym polskim tokenizerem BPE (32 768 pozycji).

Model jest **artefaktem eksperymentu badawczego**, nie narzędziem użytkowym.
Pytanie: *czy dedykowany tokenizer polski daje lepszy model niż tokenizer GPT-2 przy
tym samym budżecie treningowym?* Odpowiedź: **nie daje lepszego — daje porównywalny
2,5× taniej.**

Kod, dane i pełna metodologia: https://github.com/KateMajzel/gollem-pl

## Użycie

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="KateMajzel/GoLLeM-45M-PL")
print(pipe("W lesie mieszkał mały", max_new_tokens=50)[0]["generated_text"])
```

Domyślne parametry generacji: `do_sample=True`, `temperature=0.8`, `top_p=0.9`,
`repetition_penalty=1.15`. Dekodowanie zachłanne wpada w pętle powtórzeń — typowe dla
modeli tej wielkości. 199 nieużywanych tokenów wypełniających (32 569–32 767)
zablokowano przez `bad_words_ids`.

## Wyniki

BPB na prywatnym held-oucie (1 999 dokumentów, identyczne 2 767 440 bajtów dla
każdego modelu; średnie z 3 ziaren dla R1 i R2a):

| przebieg | tokenizer | budżet | BPB ↓ | sd | parametry | czas |
|---|---|---|---|---|---|---|
| **R1** (ten model) | polski, 32 768 | 2,96 GB | 1,2114 | 0,0100 | 42,5 M | **49,7 min** |
| R2a | GPT-2, 50 257 | 2,96 GB — te same bajty | 1,1946 | 0,0035 | 51,5 M | 122,4 min |
| R2b | GPT-2, 50 257 | 2 970 kroków — te same tokeny | 1,2756 | (n=1) | 51,5 M | 62,4 min |
| R3 | GPT-2 zero-shot | — | 2,9555 | (n=1) | 124 M | — |

Benchmarki zero-shot (log-likelihood z normalizacją PMI domenową):

| zadanie | R1 | R2a | GPT-2 | klasa większościowa | losowo |
|---|---|---|---|---|---|
| PolEmo2-IN | 47,2% | 43,8% | 20,8% | 40,0% | 25,0% |
| 8Tags | 31,5% | 29,8% | 17,8% | 16,5% | 12,5% |

### Interpretacja

Cztery niezależne pomiary (BPB, PolEmo2, 8Tags, prywatny held-out) nie wykazują
istotnej przewagi żadnego tokenizera przy równym budżecie tekstu, a ich kierunki są
niespójne. Rozstrzygająca obserwacja: **różnica BPB między dwoma ziarnami tego samego
modelu (0,0186) przewyższa różnicę między modelami o odmiennych tokenizerach (0,0124).**

Przewaga leży w koszcie. Przy wyrównanym budżecie obliczeniowym (R1 vs R2b) tokenizer
polski wygrywa o 5,30% BPB — 6,4 odchylenia. Ten sam wniosek uzyskał niezależnie zespół
Bielika v3 PL na skali 11B (arXiv 2604.10799).

### Uwaga o perplexity

PPL/token wynosi 30,7 dla R1 i 5,8 dla R2a, co sugerowałoby pięciokrotną przewagę
GPT-2. To artefakt: modele o różnych tokenizerach przewidują jednostki różnej trudności.
**Perplexity jest między nimi nieporównywalna.** Jedyną poprawną metryką jest
bits-per-byte.

## Tokenizer

Byte-level BPE, 32 568 pozycji + 200 specjalnych. 27,9% pozycji z polskimi
diakrytykami, round-trip bezstratny, zero tokenów nieosiągalnych przez merge.

| | ten tokenizer | GPT-2 |
|---|---|---|
| bajty/token (korpus) | 4,050 | 2,066 |
| bajty/token (held-out) | 4,103 | 2,134 |
| gęstość | **1,96×** | 1,00× |

## Dane treningowe

2,96 GB, 803 177 dokumentów, [SpeakLeash](https://speakleash.org/): 37% encyklopedia
i literatura, 22% fora, 27% web, 11% publicystyka i nauka, 3% teksty urzędowe.

Celowo wykluczone: tłumaczenia i napisy (translationese), tekst syntetyczny, teksty
piosenek (prawa autorskie), surowy Common Crawl. Licencje zbiorów po stronie SpeakLeash.

## Architektura

8 warstw × 8 głowic × 512 wymiarów, kontekst 1 024 tokeny, 42,5 M parametrów
(25,7 M niezanurzeniowych), 731 mln tokenów treningowych (17,2 na parametr),
AdamW lr 1e-3 → 1e-4, bfloat16, 1 epoka, 1× RTX 5080.

## Ograniczenia

- **Skala.** Model produkuje poprawną polszczyznę na poziomie zdania, ale halucynuje
  fakty i traci spójność po kilku zdaniach. Przykład z generacji: *„wieś znalazła się
  w powiecie częstochowskim, gminie Łęczna"* — obie nazwy istnieją, zestawienie jest
  fałszywe.
- **Model bazowy.** Wyłącznie kontynuacja tekstu; bez dostrajania instrukcyjnego
  i bez filtrowania bezpieczeństwa.
- **Dane z forów internetowych.** Model może odtwarzać obecne tam uprzedzenia i język.
  Nie nadaje się do zastosowań z udziałem użytkownika końcowego — w szczególności
  dzieci — bez nadzoru człowieka.
- **Jedna skala i jedna konfiguracja.** Wyniki dotyczą 42 M parametrów i 3 GB danych.
- **Rejestr zależny od miksu danych.** Przy prompcie z bajki dla dzieci model
  odpowiada językiem forów lub haseł encyklopedycznych.
- Korpus zawiera windowsowe końce linii (`\r\n`), które model odtwarza.

## Odtwarzalność

Kod, konfiguracje, skrypty ewaluacyjne i opis pięciu pułapek metodologicznych:
https://github.com/KateMajzel/gollem-pl

Bazuje na [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT, Andrej Karpathy).

## Cytowanie

```bibtex
@misc{gollem45mpl,
  title  = {GoLLeM-45M-PL: ablacja tokenizera dla polskiego modelu językowego},
  author = {Majzel-Pośpiech, Katarzyna},
  year   = {2026},
  url    = {https://huggingface.co/KateMajzel/GoLLeM-45M-PL}
}
```
