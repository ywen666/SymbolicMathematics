DATA_PREFIX=/home/ewen/data/math

python main.py \
    --exp_name ode1_eval \
    --eval_only true \
    --reload_model "save/ode1_baseline/ode1.pth" \
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
    #--reload_data "prim_fwd,prim_fwd.train,prim_fwd.valid,prim_fwd.test"  # data location
