---
language:
- pl
license: mit
library_name: tokenizers
tags:
- polish
- tokenizer
- bpe
- byte-level
---

# tokenizer-pl-32k

Byte-level BPE dla języka polskiego. **32 568 pozycji słownika + 200 tokenów
specjalnych = 32 768** (2¹⁵, wygodne dla kerneli GPU i tensor-parallel).

Tokenizer wytrenowany na polskim korpusie i zwalidowany eksperymentalnie w projekcie
[gollem-pl](https://github.com/KateMajzel/gollem-pl) — kontrolowanej ablacji
porównującej go z tokenizerem GPT-2 przy identycznym budżecie treningowym.

Model wytrenowany z tym tokenizerem:
[KateMajzel/GoLLeM-45M-PL](https://huggingface.co/KateMajzel/GoLLeM-45M-PL).

## Gęstość

| | ten tokenizer | GPT-2 |
|---|---|---|
| bajty/token — korpus mieszany PL (2,96 GB) | **4,050** | 2,066 |
| bajty/token — held-out (2,77 MB) | **4,103** | 2,134 |
| tokenów z 2,96 GB tekstu | 731 mln | 1 434 mln |
| **gęstość względna** | **1,96×** | 1,00× |

Fertility na polskiej prozie: ~1,35 tokena na słowo (5,80 znaku/token). **Uwaga:**
na realnym korpusie — z URL-ami, liczbami, resztkami formatowania — wychodzi 4,05
bajtu/token, czyli wyraźnie gorzej. Fertility należy podawać z korpusu, nie
z wyselekcjonowanych zdań.

Przykład różnicy w praktyce — słowo `niejednoznaczny` to **3 tokeny** tutaj i **9**
u GPT-2.

## Właściwości

| | |
|---|---|
| typ | byte-level BPE (`tokenizers`, format `tokenizer.json`) |
| normalizacja | brak — round-trip bezstratny |
| pre-tokenizer | w stylu GPT-4: `Isolated` + `ByteLevel(use_regex=false, add_prefix_space=false)` |
| cyfry | cięte po maks. 3 (`\p{N}{1,3}`) |
| `byte_fallback` | nie jest potrzebny (byte-level nie może wyprodukować UNK) |
| tokeny z diakrytykami | 27,9% słownika |
| tokeny nieosiągalne przez merge | 0 (256 bajtów + 32 312 merge'ów = 32 568) |
| tokeny „same białe znaki" dłuższe niż 2 zn. | 18 |
| tokeny czysto liczbowe | 409 |

Round-trip zweryfikowany m.in. na `Zażółć gęślą jaźń — «cytat» … 😀\ttab`
i wielokrotnych spacjach oraz znakach nowej linii.

## Tokeny specjalne

`<|endoftext|>` (32568), `<|begin_of_text|>`, `<|pad|>`, `<|unk|>`, tokeny czatowe,
FIM i tool-call oraz 186 pozycji `<|reserved_N|>` — zapas na przyszłe rozszerzenia
bez zmiany kształtu tablicy embeddingów.

**Ważne przy trenowaniu od zera:** jeśli korzystasz tylko z `<|endoftext|>` jako
separatora dokumentów, pozostałe tokeny specjalne nigdy nie dostaną gradientu.
Warto je wtedy zablokować przy generowaniu (`bad_words_ids`) i ustawić
`bos_token_id = eos_token_id = 32568`.

## Użycie

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("KateMajzel/tokenizer-pl-32k")
ids = tok.encode("Zażółć gęślą jaźń").ids
print(len(ids), tok.decode(ids))
```

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("KateMajzel/tokenizer-pl-32k")
```

## Co pokazał eksperyment

Ablacja na modelach 42–51 M parametrów (3 GB korpusu, 3 ziarna losowe) wykazała:

- **Przy równym budżecie tekstu tokenizer nie różnicuje jakości modelu.** Cztery
  niezależne pomiary (BPB, PolEmo2, 8Tags, prywatny held-out) nie wykazały istotnej
  różnicy wobec tokenizera GPT-2, a ich kierunki były niespójne. Różnica BPB między
  dwoma ziarnami tego samego modelu (0,0186) przewyższała różnicę między tokenizerami
  (0,0124).
- **Przy równym budżecie obliczeniowym przewaga jest jednoznaczna** — 5,30% BPB
  (6,4 odchylenia) na korzyść gęstszego tokenizera.
- **Zysk leży w koszcie:** ten sam poziom BPB osiągnięty w 2,5× krótszym czasie,
  przy 21% mniejszej liczbie parametrów.

Ten sam wniosek uzyskał niezależnie zespół Bielika v3 PL na skali 11B
([arXiv 2604.10799](https://arxiv.org/abs/2604.10799)): wymiana tokenizera na
dedykowany zachowała jakość i niemal podwoiła gęstość reprezentacji.

## Ograniczenia

- **Angielski i kod tokenizują się gorzej niż w GPT-2** — 3,0 i 2,4 znaku/token wobec
  ~4,0. To zamierzony kompromis, ale wyklucza użycie do zbiorów wielojęzycznych.
- **Korpus treningowy przechylony w stronę tekstów administracyjno-prasowych** —
  widać to w najdłuższych tokenach (`Ġniepełnosprawności`, `Ġzagospodarowania`).
  Przy modelu, który ma brzmieć potocznie, warto to uwzględnić w miksie danych.
- **Reguła apostrofów w pre-tokenizerze jest angielska** (`'(?i:[sdmt]|ll|ve|re)`) —
  na polskim martwa, ale nieszkodliwa.
- **Brak `post_processor`** — BOS/EOS nie są dodawane automatycznie, offsety nie są
  trymowane. Bez znaczenia dla modelowania języka, istotne przy NER/QA.
- **Koszt embeddingu.** Przy `d_model = 512` tablica embeddingów to 16,8 M parametrów.
  W modelu 42,5 M stanowi to 40% — warto to policzyć, zanim dobierze się rozmiar
  słownika do małego modelu.

## Licencja i cytowanie

MIT. Metodologia i kod: https://github.com/KateMajzel/gollem-pl

```bibtex
@misc{tokenizerpl32k,
  title  = {tokenizer-pl-32k: polski tokenizer BPE i jego ablacja},
  author = {Majzel-Pośpiech, Katarzyna},
  year   = {2026},
  url    = {https://huggingface.co/KateMajzel/tokenizer-pl-32k}
}
```
