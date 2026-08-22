#!/usr/bin/env python3
"""Znajdz maksymalny batch_size i zmierz realne FLOP/s na TWOJEJ karcie.

Uruchom z katalogu nanoGPT (potrzebuje model.py).

    python probe_vram.py --n-layer 12 --n-embd 768 --vocab 32768
    python probe_vram.py --n-layer 8  --n-embd 512 --vocab 32768   # tier B

Wypisuje szczytowy VRAM, tok/s, efektywne TFLOPS i MFU dla kolejnych batchy,
oraz proponuje gradient_accumulation_steps pod zadany budzet tokenow/iteracje.
"""
import argparse, time
import torch


def try_batch(bs, args, device):
    from model import GPT, GPTConfig
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    conf = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                     block_size=args.block, vocab_size=args.vocab, dropout=0.0, bias=False)
    model = GPT(conf).to(device)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    x = torch.randint(0, args.vocab, (bs, args.block), device=device)
    y = torch.randint(0, args.vocab, (bs, args.block), device=device)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for _ in range(3):  # warmup (+ kompilacja)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    ITERS = 8
    for _ in range(ITERS):
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS

    peak = torch.cuda.max_memory_allocated() / 2**30
    n_params = sum(p.numel() for p in model.parameters())
    n_nonemb = n_params - args.vocab * args.n_embd
    toks = bs * args.block
    flops = 6 * n_nonemb * toks + 12 * args.n_layer * args.n_embd * args.block * toks
    del model, opt, x, y
    torch.cuda.empty_cache()
    return peak, toks / dt, flops / dt, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--target-tokens-per-iter", type=int, default=245760)
    ap.add_argument("--peak-tflops", type=float, default=0.0,
                    help="szczytowe bf16 TFLOPS karty, do policzenia MFU (opcjonalne)")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Brak CUDA.")
    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = torch.cuda.get_device_capability()
    print(f"{name}  |  {total:.1f} GiB  |  sm_{cap[0]}{cap[1]}  |  torch {torch.__version__}")
    if cap[0] >= 12:
        print("(Blackwell — upewnij sie, ze masz build z CUDA 12.8+)")
    print(f"model: {args.n_layer}L/{args.n_embd}d vocab={args.vocab} block={args.block}"
          f"{' +compile' if args.compile else ''}\n")

    print(f"{'batch':>5} {'VRAM peak':>10} {'tok/s':>9} {'TFLOPS':>8} {'MFU':>6}  status")
    best = None
    for bs in args.batches:
        try:
            peak, tps, fps, npar = try_batch(bs, args, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{bs:>5} {'-':>10} {'-':>9} {'-':>8} {'-':>6}  OOM")
            break
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print(f"{bs:>5}  BLAD: {str(e)[:80]}")
            break
        mfu = f"{fps/1e12/args.peak_tflops*100:5.1f}%" if args.peak_tflops else "   n/a"
        head = peak / total
        print(f"{bs:>5} {peak:>9.2f}G {tps:>9,.0f} {fps/1e12:>8.1f} {mfu:>6}  "
              f"{'OK' if head < 0.90 else 'ciasno'}")
        if head < 0.90:
            best = (bs, tps, fps, npar)

    if not best:
        print("\nNic sie nie zmiescilo — zmniejsz --block do 512 albo model.")
        return
    bs, tps, fps, npar = best
    ga = max(1, round(args.target_tokens_per_iter / (bs * args.block)))
    print(f"\n>>> parametry modelu: {npar/1e6:.1f}M")
    print(f">>> ZALECANE:  batch_size = {bs}   gradient_accumulation_steps = {ga}"
          f"   (= {bs*ga*args.block:,} tok/iter)")
    print(f">>> efektywne {fps/1e12:.1f} TFLOPS -> {tps:,.0f} tok/s\n")

    print("czas treningu dla roznych budzetow (przy tej przepustowosci):")
    for label, gb, bpt in [("A smoke  300 MB", 0.3, 5.3), ("B eksper.  3 GB", 3.0, 5.3),
                           ("C final   12 GB", 12.0, 5.3), ("R2 gpt2tok 3 GB", 3.0, 2.3)]:
        toks = gb * 1e9 / bpt
        h = toks / tps / 3600
        iters = toks / (bs * ga * args.block)
        print(f"  {label:<16} {toks/1e6:>7.0f}M tok  ->  {h:>5.1f} h   max_iters = {iters:>6.0f}")


if __name__ == "__main__":
    main()
