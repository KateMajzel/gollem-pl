# TIER B — model do ABLACJI TOKENIZERA. 8L/512 ~ 45M param, 3 GB korpusu.
# Przy 566M tokenow optimum Chinchilli to ~28M param niezanurzeniowych,
# wiec 45M jest wlasciwie dobrane. 110M na tym budzecie byloby niedotrenowane.
#
# TARGET_BYTES = 3.0e9 ; bpt = 5.3 ; tokens_per_iter = 8*30*1024 = 245_760
#   3.0e9 / 5.3 / 245_760 = 2303  ->  max_iters = 2300   (~1 h)

out_dir = 'out/r1-tierB-pltok'
eval_interval = 200
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_run_name = 'r1-tierB-pltok'

dataset = 'pl_pltok'
batch_size = 8                   # mniejszy model -> wiekszy batch sie miesci
gradient_accumulation_steps = 30
block_size = 1024

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
vocab_size = 32768

learning_rate = 1e-3             # mniejszy model znosi wyzszy LR
max_iters = 2970
lr_decay_iters = 2970
min_lr = 1e-4
warmup_iters = 150
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

compile = True
dtype = 'bfloat16'
