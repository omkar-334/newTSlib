
model_name=CnDiff

cond=1
tphi=2
tphi_loss=True
normalize=True

attn_dropout=0.1
hidden_dim=32
mlp_ratio=2
n_depth=2
n_emb=2
n_heads=4
timesteps=200

filename=our_forecast
device=0


# Traffic
pred_len=168

for seq_len in 96 192 336 720 1440; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_s${seq_len}_p${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --use_cond $cond \
    --use_tphi $tphi \
    --tphi_loss $tphi_loss \
    --des 'Exp' \
    --itr 1 \
    --normalize $normalize \
    --attn_dropout $attn_dropout \
    --hidden_dim $hidden_dim \
    --mlp_ratio $mlp_ratio \
    --n_depth $n_depth \
    --n_emb $n_emb \
    --n_heads $n_heads \
    --filename $filename \
    --gpu $device \
    --timesteps $timesteps
done

