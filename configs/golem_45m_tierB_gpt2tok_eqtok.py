# R2b — ABLACJA TOKENIZERA, wariant z WYROWNANYMI TOKENAMI (nie bajtami).
#
# Po co dwa warianty R2:
#   R2a (golem_45m_tierB_gpt2tok.py) — te same BAJTY co R1 (2.96 GB).
#       Oba modele czytaja identyczny tekst. R2a zuzywa ~2.2x wiecej obliczen.
#       Zarzut krytyka: "R1 wygral, bo dostal mniej danych do przetworzenia?"
#       -> nie, dostal tyle samo tekstu; przewaga jest w gestosci reprezentacji.
#
#   R2b (ten plik) — ta sama liczba TOKENOW co R1 (731 mln, 2970 iteracji).
#       Oba modele wykonuja ~tyle samo krokow i podobna liczbe operacji.
#       R2b widzi za to tylko ~1.2 GB tekstu zamiast 2.96 GB.
#       Zarzut krytyka: "R1 wygral, bo dostal wiecej obliczen?"
#       -> nie, R2b mial ich tyle samo i przegral, bo przeczytal mniej tekstu.
#
# Dopiero R1 vs R2a vs R2b razem zamykaja obie drogi ucieczki.
# W raporcie: tabela BPB z trzema wierszami + kolumny (bajty, tokeny, czas).
#
# UWAGA co do "rownych FLOPs": czesc transformerowa jest identyczna, ale
# warstwa wyjsciowa lm_head liczy 2*d*V operacji na token, a V rozni sie
# (32768 vs 50304). R2b jest wiec o ~15-20% drozszy na token mimo tej samej
# liczby krokow. Nie udawaj, ze FLOPs sa rowne co do joty — zaraportuj
# zmierzony czas i MFU z logow (log_r1.txt, log_r2a.txt, log_r2b.txt).

out_dir = 'out/r2b-tierB-gpt2tok-eqtok'
eval_interval = 200
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_run_name = 'r2b-tierB-gpt2tok-eqtok'

dataset = 'pl_gpt2tok'        # ten sam korpus co R2a, mniej przerobiony
batch_size = 12
gradient_accumulation_steps = 20
block_size = 1024
# tokens_per_iter = 6 * 40 * 1024 = 245_760 — identycznie jak R1

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
vocab_size = 50304

learning_rate = 1e-3
max_iters = 2970              # <- DOKLADNIE tyle co R1
lr_decay_iters = 2970
min_lr = 1e-4
warmup_iters = 150
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

compile = True
dtype = 'bfloat16'
