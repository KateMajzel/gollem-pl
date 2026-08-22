# R1 — GoLLeM, wlasny tokenizer PL (32768).  TIER C (finalny) na RTX 5080 16GB.
# torchrun --standalone --nproc_per_node=1 train.py configs/golem_110m_pltok.py
#
# BUDZET LICZONY W BAJTACH KORPUSU, nie w tokenach — inaczej porownanie z R2
# jest nieuczciwe (tokenizer PL pakuje ~2.3x wiecej tekstu na token).
#
#   bytes_per_token  <- data/pl_pltok/meta.json  (mieszany PL: ~5.3)
#   tokens_per_iter  = batch_size * grad_accum * block_size
#   max_iters        = TARGET_BYTES / bytes_per_token / tokens_per_iter
#
# TARGET_BYTES = 12.0e9 ; bpt = 5.3 ; tokens_per_iter = 4*60*1024 = 245_760
#   12.0e9 / 5.3 / 245_760 = 9214  ->  max_iters = 9200   (~9-12 h)

out_dir = 'out/r1-pltok'
eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = False   # zapisuj tylko przy poprawie val loss

wandb_log = False
wandb_project = 'golem'
wandb_run_name = 'r1-pltok'

dataset = 'pl_pltok'             # data/pl_pltok/{train,val}.bin

# --- dopasowane do 16 GB VRAM ---
# domyslne nanoGPT batch_size=12 = natychmiastowy OOM na tej karcie.
# Akumulacja gradientu daje ten sam efekt matematyczny co duzy batch.
batch_size = 4                   # potwierdz przez probe_vram.py
gradient_accumulation_steps = 60
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
vocab_size = 32768               # 32568 slownika + 200 tokenow specjalnych

learning_rate = 6e-4
max_iters = 9200
lr_decay_iters = 9200
min_lr = 6e-5
warmup_iters = 300
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

compile = True                   # jesli sypie sie na sm_120 -> False (-25% szybkosci)
dtype = 'bfloat16'
