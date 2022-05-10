DATA_PREFIX=/home/ewen/data/math

#export NGPU=2; 
#python -m torch.distributed.launch --nproc_per_node=$NGPU main.py
#python main.py \
#CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 main.py \
CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name ode1_train \
    --fp16 true \
    --amp 2 \
    --tasks "ode1" \
    --reload_data "ode1,${DATA_PREFIX}/ode1.train,${DATA_PREFIX}/ode1.valid,${DATA_PREFIX}/ode1.test" \
    --reload_size 50000 \
    --emb_dim 1024 \
    --n_enc_layers 6 \
    --n_dec_layers 6 \
    --n_heads 8 \
    --optimizer "adam,lr=0.0001" \
    --batch_size 16 \
    --epoch_size 300000 \
    --contra_coeff 0.01 \
    --validation_metrics valid_ode1_acc
