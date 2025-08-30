
model_name=CnDiff
des=new
filename=imputation_sweep

device=1
tphi=1
cond=1
tphi_loss=True
normalize=True


attn_dropout=0.1
mlp_ratio=2
n_emb=2

# Ettm1
for rate in 0.125 0.25; do
  for timesteps in 100 200; do
    for hidden_dim in 32 64 128; do
      for n_depth in 2 4; do
        for n_heads in 4 8; do
          python -u run.py \
            --task_name imputation \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id ETTm1_mask_$rate \
            --mask_rate $rate \
            --model $model_name \
            --data custom \
            --features M \
            --pred_len 96 \
            --c_out 7 \
            --gpu $device \
            --d_model 96 \
            --use_tphi $tphi \
            --tphi_loss $tphi_loss \
            --attn_dropout $attn_dropout \
            --hidden_dim $hidden_dim \
            --mlp_ratio $mlp_ratio \
            --n_depth $n_depth \
            --n_emb $n_emb \
            --n_heads $n_heads \
            --timesteps $timesteps \
            --filename $filename \
            --normalize $normalize \
            --des $des  \
            --use_cond $cond
        done
      done
    done
  done
done