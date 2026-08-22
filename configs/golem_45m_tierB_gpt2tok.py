# R2 — ABLACJA TOKENIZERA. Identyczny model i korpus co golem_45m_tierB.py,
# tokenizer GPT-2 (50257). To JEDYNE porownanie, ktore uczciwie mowi,
# czy Twoj tokenizer polski cokolwiek daje.
#
# TEN SAM budzet bajtow (3.0 GB). Poniewaz bpt(gpt2, PL) ~ 2.3, wychodzi
# ~2.3x wiecej iteracji i ~2.3x wiecej FLOPs. To NIE jest blad konfiguracji —
# to jest wynik i nalezy go zaraportowac (rowne dane = wyzszy koszt).
#
#   3.0e9 / 2.3 / 245_760 = 5308  ->  max_iters = 5300   (~2.5 h)
#
# Wieksze logity (50304 vs 32768) -> mniejszy batch, ta sama liczba tok/iter.

out_dir = 'out/r2-tierB-gpt2tok'
eval_interval = 400
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_run_name = 'r2-tierB-gpt2tok'

dataset = 'pl_gpt2tok'           # TEN SAM korpus, ten sam split val, inny tokenizer
batch_size = 6
gradient_accumulation_steps = 40
block_size = 1024

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
vocab_size = 50304               # 50257 zaokraglone w gore do wielokrotnosci 64

learning_rate = 1e-3
max_iters = 5300
lr_decay_iters = 5300
min_lr = 1e-4
warmup_iters = 150
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

compile = True
dtype = 'bfloat16'

# WARIANT B (druga os porownania): ustaw max_iters = 2300, zeby wyrownac FLOPs
# zamiast bajtow. Wtedy R2 widzi 2.3x mniej tekstu. Zrob oba i pokaz
# BPB vs bajty ORAZ BPB vs FLOPs — dopiero te dwa wykresy razem sa uczciwe.
