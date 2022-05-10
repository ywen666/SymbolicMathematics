DATA_PREFIX=/scratch1/08401/ywen/data/math

python main.py \
    --exp_name ode1_eval \
    --eval_only true \
    --reload_model "dumped/ode1_train/4310767/best-valid_ode1_acc.pth" \
    --tasks "ode1" \
    --reload_data "ode1,${DATA_PREFIX}/ode1.train,${DATA_PREFIX}/ode1.valid,${DATA_PREFIX}/ode1.test" \
    --emb_dim 1024 \
    --n_enc_layers 6 \
    --n_dec_layers 6 \
    --n_heads 8 \
    --beam_eval true \
    --beam_size 1 \
    --beam_length_penalty 1.0 \
    --beam_early_stopping 1 \
    --eval_verbose 1 \
    --eval_verbose_print false
