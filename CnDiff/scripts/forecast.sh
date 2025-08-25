
model_name=CnDiff
cond=1
tphi=1
tphi_loss=False
normalize=True

attn_dropout=0.1
hidden_dim=256
mlp_ratio=1
n_depth=3
n_emb=2
n_heads=8
timesteps=200

seq_len=192

filename=forecast


# Electricity
pred_len=168

for seq_len in 96 192 336 720 1440; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_s${seq_len}_p${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs 100 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --timesteps $timesteps
done


# ETTh1
pred_len=168

for seq_len in 96 192 336 720 1440; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_s${seq_len}_p${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs 100 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
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
    --timesteps $timesteps
done

# Weather
pred_len=672

for seq_len in 96 192 336 720 1440; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_s${seq_len}_p${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs 100 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
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
    --timesteps $timesteps
done
