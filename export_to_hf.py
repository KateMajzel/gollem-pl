#!/usr/bin/env python3
"""nanoGPT checkpoint -> katalog HF (GPT2LMHeadModel) + tokenizer + config.

UWAGA na transpozycje: nanoGPT uzywa nn.Linear (weight [out, in]), a HF GPT-2
uzywa Conv1D (weight [in, out]). Dotyczy c_attn, c_proj, c_fc. Bez transpozycji
model "zaladuje sie" poprawnie i bedzie generowal smieci.

Skrypt weryfikuje konwersje porownujac logity obu modeli na losowym wejsciu.

    python export_to_hf.py --ckpt out/r1-pltok/ckpt.pt --tokenizer tokenizer.json \
        --out hf/GoLLeM-110M-PL-v2
"""
import argparse, json, os, shutil, sys
import torch


TRANSPOSE_SUFFIXES = (".attn.c_attn.weight", ".attn.c_proj.weight",
                      ".mlp.c_fc.weight", ".mlp.c_proj.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True, help="sciezka do tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    from transformers import GPT2Config, GPT2LMHeadModel

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    margs = ck["model_args"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    sd.pop("lm_head.weight", None)          # tied z wte
    sd = {k: v for k, v in sd.items() if not k.endswith(".attn.bias")}  # bufor maski

    cfg = GPT2Config(
        vocab_size=margs["vocab_size"],
        n_positions=margs["block_size"],
        n_ctx=margs["block_size"],
        n_embd=margs["n_embd"],
        n_layer=margs["n_layer"],
        n_head=margs["n_head"],
        bos_token_id=None, eos_token_id=None,   # uzupelnione nizej z tokenizera
    )
    hf = GPT2LMHeadModel(cfg)
    hf_sd = hf.state_dict()

    new = {}
    for k, v in sd.items():
        if any(k.endswith(s) for s in TRANSPOSE_SUFFIXES):
            v = v.t().contiguous()
        new[k] = v
    # nanoGPT z bias=False nie ma biasow LayerNorm/Linear; HF ich wymaga -> zera
    for k in hf_sd:
        if k not in new and k != "lm_head.weight":
            if k.endswith(".bias"):
                new[k] = torch.zeros_like(hf_sd[k])
            else:
                raise SystemExit(f"Brak wagi w checkpoincie: {k}")

    missing, unexpected = hf.load_state_dict(new, strict=False)
    unexpected = [u for u in unexpected if u != "lm_head.weight"]
    if unexpected:
        raise SystemExit(f"Nieoczekiwane klucze: {unexpected}")
    hf.tie_weights()
    hf.eval()

    # --- weryfikacja: logity nanoGPT vs HF ---
    # UWAGA: porownujemy blad WZGLEDNY, nie absolutny. nanoGPT liczy uwage przez
    # scaled_dot_product_attention (flash), HF klasycznie — inna kolejnosc operacji
    # zmiennoprzecinkowych daje rozbieznosc rzedu 1e-4 * skala_logitow nawet przy
    # w pelni poprawnej konwersji. Prog absolutny dawalby falszywe alarmy dla
    # modeli wytrenowanych (duze logity). Dodatkowo sprawdzamy zgodnosc argmax,
    # bo to ona decyduje o generacji.
    if not a.no_verify:
        try:
            from model import GPT, GPTConfig
            ng = GPT(GPTConfig(**margs))
            ng.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
            ng.eval()
            torch.manual_seed(0)
            x = torch.randint(0, margs["vocab_size"], (1, 64))
            with torch.no_grad():
                a_ = ng(x)[0][:, -1, :]        # nanoGPT bez targets: ostatnia pozycja
                b_ = hf(x).logits[:, -1, :]
            absd = (a_ - b_).abs().max().item()
            scale = a_.abs().max().item()
            rel = absd / max(scale, 1e-9)
            top1 = (a_.argmax(-1) == b_.argmax(-1)).float().mean().item()
            print(f"weryfikacja: blad wzgledny {rel:.2e}  (abs {absd:.2e}, skala logitow {scale:.1f}), "
                  f"zgodnosc top-1 {top1:.3f}")
            if rel > 1e-3 or top1 < 1.0:
                raise SystemExit("KONWERSJA NIEPOPRAWNA — sprawdz transpozycje wag.")
            print("  -> OK (rozbieznosc na poziomie precyzji float32)")
        except ImportError:
            print("(pominieto weryfikacje: brak model.py z nanoGPT na sciezce)")

    os.makedirs(a.out, exist_ok=True)

    # tokenizer + wiring tokenow specjalnych
    shutil.copy(a.tokenizer, os.path.join(a.out, "tokenizer.json"))
    tj = json.load(open(a.tokenizer, encoding="utf-8"))
    ids = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    eos = ids.get("<|endoftext|>")
    cfg.eos_token_id = eos
    cfg.bos_token_id = ids.get("<|begin_of_text|>", eos)
    cfg.pad_token_id = ids.get("<|pad|>", eos)
    hf.config = cfg

    json.dump({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": margs["block_size"],
        "bos_token": "<|begin_of_text|>" if "<|begin_of_text|>" in ids else "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>" if "<|pad|>" in ids else "<|endoftext|>",
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
    }, open(os.path.join(a.out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=2)

    hf.save_pretrained(a.out, safe_serialization=True)
    n = sum(p.numel() for p in hf.parameters())
    print(f"zapisano -> {a.out}  ({n/1e6:.1f}M parametrow, vocab={cfg.vocab_size})")
    print("dopisz jeszcze README.md (karta modelu) i wypchnij:")
    print(f"  huggingface-cli upload <user>/<repo> {a.out} .")


if __name__ == "__main__":
    main()
